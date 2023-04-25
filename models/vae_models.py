import tensorflow as tf
from tensorflow import keras
import sys


class VAEEncoder(keras.Model):
    """ Creates an encoder model for a Variational Autoencoder

    The encoder model is a multi-layer perceptron with a final layer that outputs the mean and log variance of the
    latent representation. Each of the hidden layers is a dense layer with a SELU activation function, and is
    regularized by a l2 penalty. The final layer is a dense layer with a linear activation function. Additionally,
    the output of the hidden layers is used to predict the class of the input data.
    """
    def __init__(self, hidden_dim, latent_dim, activation='relu', initializer='glorot_uniform', dropout_rate=0.2):
        super(VAEEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.initializer = initializer
        self.hidden_layers = []
        for h_dim in self.hidden_dim:
            self.hidden_layers.append(keras.layers.Dense(
                h_dim,
                activation=self.activation,
                kernel_initializer=self.initializer
            ))
        self.latent_layer = keras.layers.Dense(
            self.latent_dim,
            activation='linear',
            kernel_initializer=self.initializer)

    def call(self, x, training=False, mask=None):
        for layer in self.hidden_layers:
            x = layer(x)
            x = keras.layers.Dropout(self.dropout_rate)(x)
        mu = self.latent_layer(x)
        log_var = self.latent_layer(x)

        # Sample from latent distribution
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = mu + tf.exp(0.5 * log_var) * epsilon
        return mu, log_var, z

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "activation": self.activation,
            "initializer": self.initializer,
            "dropout_rate": self.dropout_rate
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["hidden_dim"], config["latent_dim"], config["activation"], config["initializer"],
                   config["dropout_rate"])


class VAEDecoder(keras.Model):
    """Creates a decoder model for a Variational Autoencoder

    The decoder model is a multi-layer perceptron with a final layer that outputs the mean of the input representation.
    Each of the hidden layers is a dense layer with a SELU activation function, and is regularized by a l2 penalty.
    The final layer is a dense layer with a linear activation function, and is initialized with a glorot uniform
    initializer.
    """
    def __init__(self, hidden_dim, output_dim, activation='relu', initializer='glorot_uniform', dropout_rate=0.2):
        super(VAEDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.initializer = initializer
        self.dropout_rate = dropout_rate
        self.hidden_layers = []
        for h_dim in self.hidden_dim:
            self.hidden_layers.append(keras.layers.Dense(
                h_dim,
                activation=self.activation,
                kernel_initializer=self.initializer
            ))
        self.output_layer = tf.keras.layers.Dense(
            self.output_dim,
            activation='softmax',
            kernel_initializer=self.initializer
        )

    def call(self, x, training=False, mask=None):
        for layer in self.hidden_layers:
            x = layer(x)
            x = keras.layers.Dropout(self.dropout_rate)(x)
        output = self.output_layer(x)
        return output

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "initializer": self.initializer,
            "dropout_rate": self.dropout_rate
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["hidden_dim"], config["output_dim"], config["activation"], config["initializer"],
                   config["dropout_rate"])


def calc_kl_loss(mu, log_var):
    loss = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss


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
        self.optimizer = None

    def compile(
            self,
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            reconstruction_loss_fn=tf.keras.losses.MeanSquaredError(),
            kl_loss_fn=calc_kl_loss,
            **kwargs
    ):
        super(VAE, self).compile(**kwargs)
        self.optimizer = optimizer
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.kl_loss_fn = kl_loss_fn

    def train_step(self, batch_data):
        x, y = batch_data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, x_reconstructed = self(batch_data, training=True)
            reconstruction_loss = self.reconstruction_loss_fn(x, x_reconstructed)
            kl_loss = tf.reduce_mean(self.kl_loss_fn(z_mean, z_log_var))

            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "lr": self.optimizer.lr
        }

    def test_step(self, batch_data):
        x, y = batch_data
        z_mean, z_log_var, z, x_reconstructed = self(batch_data, training=False)
        reconstruction_loss = self.reconstruction_loss_fn(x, x_reconstructed)
        kl_loss = tf.reduce_mean(self.kl_loss_fn(z_mean, z_log_var))
        total_loss = reconstruction_loss + kl_loss
        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }

    def call(self, data, training=False, mask=None):
        x, y = data
        z_mean, z_log_var, z = self.encoder(x, training=training)
        x_reconstructed = self.decoder(z, training=training)
        return z_mean, z_log_var, z, x_reconstructed

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


#%%
