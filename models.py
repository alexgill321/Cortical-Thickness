import tensorflow as tf
from tensorflow import keras


# noinspection PyMethodOverriding
class Encoder(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Define the input layer
        self.input_layer = keras.Input(shape=(self.input_dim,))

        # Define the latent representation layer
        self.latent_layer = keras.layers.Dense(self.latent_dim, activation='relu')

        self.model = self.build_graph()

    def build_graph(self):
        # Connect the layers
        inputs = self.input_layer
        x = inputs
        for n_neurons in self.hidden_dim:
            x = keras.layers.Dense(n_neurons)(x)
            x = keras.layers.LeakyReLU()(x)
        latent = self.latent_layer(x)

        # Create a Keras model for the encoder
        return keras.Model(inputs=inputs, outputs=latent, name='encoder')

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(config["input_dim"], config["hidden_dim"], config["latent_dim"])


# noinspection PyMethodOverriding
class Decoder(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Define the input layer
        self.input_layer = keras.Input(shape=(self.latent_dim,))

        # Define the output representation layer
        self.output_layer = keras.layers.Dense(self.input_dim, activation='linear')

        self.model = self.build_graph()

    def build_graph(self):
        # Connect the layers
        inputs = self.input_layer
        x = inputs
        for n_neurons in self.hidden_dim:
            x = keras.layers.Dense(n_neurons)(x)
            x = keras.layers.LeakyReLU()(x)
        output = self.output_layer(x)

        # Create a Keras model for the decoder
        return keras.Model(inputs=inputs, outputs=output, name='decoder')

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(config["input_dim"], config["hidden_dim"], config["latent_dim"])


# noinspection PyMethodOverriding
class Discriminator(tf.keras.Model):
    def __init__(self, latent_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Define the input layer
        self.input_layer = keras.Input(shape=(self.latent_dim,))

        # Define the output representation layer
        self.output_layer = keras.layers.Dense(1)

        self.model = self.build_graph()

    def build_graph(self):
        # Connect the layers
        inputs = self.input_layer
        x = inputs
        for n_neurons in self.hidden_dim:
            x = keras.layers.Dense(n_neurons)(x)
            x = keras.layers.LeakyReLU()(x)
        output = self.output_layer(x)

        # Create a Keras model for the discriminator
        return keras.Model(inputs=inputs, outputs=output, name='discriminator')

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(config["hidden_dim"], config["latent_dim"])


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

    def call(self, inputs):
        x, y = inputs
        encoder_output = self.encoder(x, training=False)
        decoder_output = self.decoder(tf.concat([encoder_output, y], axis=1), training=False)
        return decoder_output

    def get_config(self):
        return {
            "encoder_config": self.encoder.get_config(),
            "decoder_config": self.decoder.get_config(),
            "discriminator_config": self.discriminator.get_config(),
            "z_dim": self.z_dim,
        }

    @classmethod
    def from_config(cls, config):
        encoder = Encoder.from_config(config["encoder_config"])
        decoder = Decoder.from_config(config["decoder_config"])
        discriminator = Discriminator.from_config(config["discriminator_config"])
        z_dim = config["z_dim"]
        return cls(encoder, decoder, discriminator, z_dim)
# %%
