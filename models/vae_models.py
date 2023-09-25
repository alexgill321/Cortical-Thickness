import tensorflow as tf
from tensorflow import keras


def create_vae_encoder(input_dim, hidden_dim, latent_dim, activation='relu', initializer='glorot_uniform',
                       dropout_rate=0.2):
    """Creates an encoder for a Variational Autoencoder

    Args:
        input_dim (int): The dimension of the input data
        hidden_dim (list): A list of the hidden dimensions for the encoder
        latent_dim (int): The dimensionality of the latent space
        activation (str): The activation function to use for the hidden layers
        initializer (str): The initializer to use for the hidden layers
        dropout_rate (float): The dropout rate to use for the hidden layers

    Returns: The encoder model with the specified layers and parameters
    """
    inputs = keras.layers.Input(shape=(input_dim,))
    x = inputs
    for h_dim in hidden_dim:
        x = keras.layers.Dense(h_dim, kernel_initializer=initializer)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
    mu = keras.layers.Dense(
        latent_dim,
        activation='linear',
        kernel_initializer=initializer)(x)
    log_var = keras.layers.Dense(
        latent_dim,
        activation='linear',
        kernel_initializer=initializer)(x)

    model = keras.Model(inputs, [mu, log_var], name='VAEEncoder')
    return model


def create_vae_decoder(latent_dim, hidden_dim, output_dim, activation='relu', initializer='glorot_uniform',
                       dropout_rate=0.2):
    """Creates a decoder for a Variational Autoencoder

    Args:
        latent_dim (int): The dimensionality of the latent space
        hidden_dim (list): A list of the hidden dimensions for the decoder
        output_dim (int): The dimension of the output data
        activation (str): The activation function to use for the hidden layers
        initializer (str): The initializer to use for the hidden layers
        dropout_rate (float): The dropout rate to use for the hidden layers

    Returns: The decoder model with the specified layers and parameters
    """
    # Define the input layer
    decoder_input = keras.layers.Input(shape=(latent_dim,), name='decoder_input')

    # Add the hidden layers
    x = decoder_input
    # Reverse the order of the hidden layers
    hidden_dim = hidden_dim[::-1]
    # Add the hidden layers
    for h_dim in hidden_dim:
        x = keras.layers.Dense(
            h_dim,
            kernel_initializer=initializer)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation)(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    # Add the output layer
    decoder_output = keras.layers.Dense(output_dim, activation='linear', kernel_initializer=initializer)(x)

    # Define the model
    decoder = keras.Model(decoder_input, decoder_output, name='VAEDecoder')

    return decoder


def calc_kl_loss(mu, log_var):
    """Calculates the KL divergence loss for a Variational Autoencoder

    This function calculates the KL divergence loss for a Variational Autoencoder. The KL divergence loss is the
    difference between the latent distribution and a standard normal distribution.

    Args:
        mu (tensor): The mean of the latent distribution
        log_var (tensor): The log variance of the latent distribution

    Returns: The kl divergence loss, or the difference between the latent distribution and a standard normal
    distribution
    """
    loss = -0.5 * (1 + log_var - tf.square(mu) - tf.exp(log_var))
    loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss


def r2_score(y_true, y_predicted):
    """Calculates the R2 score

    Args:
        y_true (tensor): The true values
        y_predicted (tensor): The predicted values

    Returns: The R2 score
    """
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_predicted)), axis=1)
    mean = tf.reduce_mean(y_true, axis=1, keepdims=True)
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, mean)), axis=1)
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2


def r2_feat_score(y_true, y_predicted):
    """Calculate the Measurement based R2 score for features across subjects

    Args:
        y_true (tensor): True values (per subject)
        y_predicted (tensor): The predicted values (per subject)

    Returns: The R2 score calculated across subjects
    """
    # Invert the shape of the tensors to be (subject, feature)
    y_true = tf.transpose(y_true)
    y_predicted = tf.transpose(y_predicted)

    # Calculate the residual and total sums of squares
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_predicted)), axis=1)
    mean = tf.reduce_mean(y_true, axis=1, keepdims=True)
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, mean)), axis=1)
    r2 = tf.subtract(1.0, tf.divide(residual, total))
    return r2


class VAE(keras.Model):
    """ Variational Autoencoder Model

    This class implements a Variational Autoencoder model. The model generates a latent representation of the input
    data, and then reconstructs the input data from the latent representation. The model is trained to minimize the
    reconstruction loss and the KL divergence between the latent distribution and a standard normal distribution.

    Args:
        encoder (VAEEncoder): The encoder model
        decoder (VAEDecoder): The decoder model

    Attributes:
        encoder (VAEEncoder): The encoder model
        decoder (VAEDecoder): The decoder model
        reconstruction_loss_fn (function): The function to use to calculate the reconstruction loss
        kl_loss_fn (function): The function to use to calculate the KL divergence loss
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to use to train the model
        beta (float): The weight to use for the KL divergence loss
        cov (bool): Whether or not to use the covariance matrix as input to the decoder
    """
    def __init__(
            self,
            encoder,
            decoder,
            beta=1,
            cov=True
    ):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reconstruction_loss_fn = None
        self.kl_loss_fn = None
        self.optimizer = None
        self.beta = beta
        self.cov = cov

    def compile(
            self,
            optimizer=None,
            reconstruction_loss_fn=tf.keras.losses.MeanSquaredError(),
            kl_loss_fn=calc_kl_loss,
            **kwargs
    ):
        super(VAE, self).compile(**kwargs)
        self.optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.kl_loss_fn = kl_loss_fn

    def train_step(self, batch_data):
        x, y = batch_data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, x_reconstructed = self(batch_data, training=True)
            reconstruction_loss = self.reconstruction_loss_fn(x, x_reconstructed)
            kl_loss = tf.reduce_mean(self.kl_loss_fn(z_mean, z_log_var))
            total_loss = reconstruction_loss + self.beta * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "lr": self.optimizer.lr,
            "beta": self.beta
        }

    def test_step(self, batch_data):
        x, y = batch_data
        z_mean, z_log_var, z, x_reconstructed = self(batch_data, training=False)
        reconstruction_loss = self.reconstruction_loss_fn(x, x_reconstructed)
        kl_loss = tf.reduce_mean(self.kl_loss_fn(z_mean, z_log_var))
        total_loss = reconstruction_loss + kl_loss
        x = tf.cast(x, dtype=tf.float32)
        r2 = r2_score(x, x_reconstructed)
        r2_feat = r2_feat_score(x, x_reconstructed)
        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "r2": r2,
            "r2_feat": r2_feat
        }

    def call(self, data, training=False, mask=None):
        x, y = data
        if self.cov:
            x_cov = tf.concat([x, y], axis=-1)
            z_mean, z_log_var = self.encoder(x_cov, training=training)
        else:
            z_mean, z_log_var = self.encoder(x, training=training)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        if self.cov:
            z_cov = tf.concat([z, y], axis=-1)
            x_reconstructed = self.decoder(z_cov, training=training)
        else:
            x_reconstructed = self.decoder(z, training=training)
        return z_mean, z_log_var, z, x_reconstructed

    def get_config(self):
        return {
            "encoder": self.encoder.get_config(),
            "decoder": self.decoder.get_config(),
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        encoder = keras.models.model_from_config(config["encoder"], custom_objects=custom_objects)
        decoder = keras.models.model_from_config(config["decoder"], custom_objects=custom_objects)
        return cls(encoder, decoder)


#%%
