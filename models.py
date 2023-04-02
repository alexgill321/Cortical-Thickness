import tensorflow as tf
from tensorflow import keras


def create_encoder(n_features, h_dim, z_dim):
    """Creates the encoder."""
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    encoded = keras.layers.Dense(z_dim)(x)
    model = keras.Model(inputs=inputs, outputs=encoded)
    return model


def create_decoder(encoded_dim, n_features, h_dim):
    """Creates the decoder."""
    encoded = keras.Input(shape=(encoded_dim,))
    x = encoded
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    reconstruction = keras.layers.Dense(n_features, activation='linear')(x)
    model = keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def create_discriminator(z_dim, h_dim):
    """Creates the discriminator."""
    z_features = keras.Input(shape=(z_dim,))
    x = z_features
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    prediction = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=z_features, outputs=prediction)
    return model


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_fake + loss_real


def autoencoder_loss(x, output):
    return tf.keras.losses.MeanSquaredError()(x, output)


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# noinspection PyMethodOverriding
class AAE(keras.Model):

    def __init__(
            self,
            encoder,
            decoder,
            discriminator,
            z_dim
    ):
        super(AAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.accuracy = tf.keras.metrics.BinaryAccuracy()

    def compile(
            self,
            encoder_optimizer,
            generator_optimizer,
            discriminator_optimizer,
            generator_loss_fn,
            autoencoder_loss_fn,
            discriminator_loss_fn,
            **kwargs
    ):
        super(AAE, self).compile(**kwargs)
        self.autoencoder_optimizer = encoder_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.generator_loss_fn = generator_loss_fn
        self.autoencoder_loss_fn = autoencoder_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn


    def train_step(self, batch_data):
        batch_x, batch_y = batch_data
        with tf.GradientTape() as ae_tape, tf.GradientTape() as dc_tape, tf.GradientTape() as gen_tape:
            # -------------------------------------------------------------------------------------------------------------
            # Autoencoder
            encoder_output = self.encoder(batch_x, training=True)
            decoder_output = self.decoder(tf.concat([encoder_output, batch_y], axis=1), training=True)

            # Autoencoder loss
            ae_loss = self.autoencoder_loss_fn(batch_x, decoder_output)

            # -------------------------------------------------------------------------------------------------------------
            # Discriminator
            real_distribution = tf.random.normal([batch_x.shape[1], self.z_dim], mean=0.0, stddev=1.0)
            encoder_output = self.encoder(batch_x, training=True)

            dc_real = self.discriminator(real_distribution, training=True)
            dc_fake = self.discriminator(encoder_output, training=True)

            # Discriminator Loss
            dc_loss = self.discriminator_loss_fn(dc_real, dc_fake)

            # Discriminator Acc
            dc_acc = self.accuracy(tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                                   tf.concat([dc_real, dc_fake], axis=0))

            # -------------------------------------------------------------------------------------------------------------
            # Generator (Encoder)
            encoder_output = self.encoder(batch_x, training=True)
            dc_fake = self.discriminator(encoder_output, training=True)

            # Generator loss
            gen_loss = self.generator_loss_fn(dc_fake)

        ae_grads = ae_tape.gradient(ae_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.autoencoder_optimizer.apply_gradients(
            zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        dc_grads = dc_tape.gradient(dc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(dc_grads, self.discriminator.trainable_variables))

        gen_grads = gen_tape.gradient(gen_loss, self.encoder.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.encoder.trainable_variables))

        return {
            "ae_loss": ae_loss,
            "dc_loss": dc_loss,
            "dc_acc": dc_acc,
            "gen_loss": gen_loss
        }
#%%
