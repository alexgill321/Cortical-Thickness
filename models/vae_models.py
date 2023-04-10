import tensorflow as tf
from tensorflow import keras


def reparameterize(mean, log_var):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(log_var * 0.5) + mean


class VAEEncoder(keras.Model):
    """ Creates an encoder model for a Variational Autoencoder

    The encoder model is a multi-layer perceptron with a final layer that outputs the mean and log variance of the
    latent representation. Each of the hidden layers is a dense layer with a SELU activation function, and is
    regularized by a l2 penalty. The final layer is a dense layer with a linear activation function. Additionally,
    the output of the hidden layers is used to predict the class of the input data.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, y_dim, l2_reg=1e-3):
        super(VAEEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.y_dim = y_dim
        self.l2_reg = l2_reg
        self.input_layer = keras.Input(shape=(self.input_dim,))
        self.hidden_layers = []
        self.y_layer = keras.layers.Dense(
            y_dim,
            activation='softmax')
        self.latent_layer = keras.layers.Dense(
            self.latent_dim * 2,
            activation=None,
            kernel_regularizer=keras.regularizers.l2(l2_reg))
        for i in range(len(self.hidden_dim)):
            self.hidden_layers.append(keras.layers.Dense(
                self.hidden_dim[i],
                activation='selu',
                kernel_initializer='lecun_normal',
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ))

    def call(self, inputs, training=False, mask=None):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        z = self.latent_layer(x)
        z_mean, z_log_var = tf.split(z, num_or_size_splits=2, axis=1)
        z = reparameterize(z_mean, z_log_var)
        y = self.y_layer(x)
        return z_mean, z_log_var, z, y

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "y_dim": self.y_dim,
            "l2_reg": self.l2_reg
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["hidden_dim"], config["latent_dim"], config["y_dim"], config["l2_reg"])


class VAEDecoder(keras.Model):
    """Creates a decoder model for a Variational Autoencoder

    The decoder model is a multi-layer perceptron with a final layer that outputs the mean of the input representation.
    Each of the hidden layers is a dense layer with a SELU activation function, and is regularized by a l2 penalty.
    The final layer is a dense layer with a linear activation function, and is initialized with a glorot uniform
    initializer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, l2_reg=1e-3):
        super(VAEDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l2_reg = l2_reg
        self.input_layer = keras.Input(shape=(input_dim,))
        self.dense_layers = []
        for i in range(len(self.hidden_dim)):
            self.dense_layers.append(keras.layers.Dense(
                self.hidden_dim[i],
                activation='selu',
                kernel_initializer='lecun_normal',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            ))
        self.dense_layers.append(tf.keras.layers.Dense(
            self.output_dim,
            activation='linear',
            kernel_initializer='glorot_uniform'
        ))

    def call(self, inputs, training=False, mask=None):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "l2_reg": self.l2_reg
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["hidden_dim"], config["input_dim"], config["output_dim"], config["l2_reg"])


class VAE(keras.Model):
    """
    Class for creation, training, and evaluation of a Variational Autoencoder model
    """
    def __init__(
            self,
            encoder,
            decoder,
    ):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_fn = None
        self.kl_loss_fn = None
        self.y_loss_fn = None
        self.optimizer = None

    def compile(
            self,
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            reconstruction_loss_fn=tf.keras.losses.MeanSquaredError(),
            kl_loss_fn=lambda x, mu, log_var: -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var),
                                                                   axis=1),
            y_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            **kwargs
    ):
        super(VAE, self).compile(**kwargs)
        self.optimizer = optimizer
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.kl_loss_fn = kl_loss_fn

    def train_step(self, batch_data):
        x, y = batch_data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, y_hat, x_reconstructed = self(batch_data, training=True)

            reconstruction_loss = self.reconstruction_loss_fn(x, x_reconstructed)
            kl_loss = tf.reduce_mean(self.kl_loss_fn(z, z_mean, z_log_var))
            y_loss = self.y_loss_fn(y, y_hat)

            total_loss = reconstruction_loss + kl_loss + y_loss

        grads = tape.gradient(total_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "y_loss": y_loss,
            "total_loss": total_loss,
            "lr": self.optimizer.lr
        }

    def eval(self, data):
        data = data.batch(len(data))
        batch = next(iter(data))
        _, _, _, y_hat, output = self(batch)
        x, y = batch
        results = tf.keras.metrics.MSE(x, output)
        # TODO: Return y loss as y_results
        return results

    def call(self, data, training=False, mask=None):
        x, y = data
        z_mean, z_log_var, z, y_hat = self.encoder(x, training=training)
        x_reconstructed = self.decoder(tf.concat([z, y], axis=1), training=training)
        return z_mean, z_log_var, z, y_hat, x_reconstructed

    def get_config(self):
        return {
            "encoder_config": self.encoder.get_config(),
            "decoder_config": self.decoder.get_config(),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        encoder = VAEEncoder.from_config(config["encoder_config"], custom_objects=custom_objects)
        decoder = VAEDecoder.from_config(config["decoder_config"], custom_objects=custom_objects)
        return cls(encoder, decoder)
