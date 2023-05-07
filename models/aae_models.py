import tensorflow as tf
from tensorflow import keras


class AAEEncoder(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AAEEncoder, self).__init__()
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

    def call(self, inputs, training=False, mask=None):
        return self.model(inputs)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["input_dim"], config["hidden_dim"], config["latent_dim"])


class AAEDecoder(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, hidden_dim):
        super(AAEDecoder, self).__init__()
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

    def call(self, inputs, training=False, mask=None):
        return self.model(inputs)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["input_dim"], config["hidden_dim"], config["latent_dim"])


class AAEDiscriminator(tf.keras.Model):
    def __init__(self, latent_dim, hidden_dim):
        super(AAEDiscriminator, self).__init__()
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

    def call(self, inputs, training=False, mask=None):
        return self.model(inputs)

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["hidden_dim"], config["latent_dim"])


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_fake + loss_real


class AAEOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        enc_optimizer=tf.keras.optimizers.Adam(),
        gen_optimizer=tf.keras.optimizers.Adam(),
        disc_optimizer=tf.keras.optimizers.Adam(),
        name='AAEOptimizer',
        **kwargs
    ):
        super().__init__(
            name=name,
            **kwargs
        )
        self.encoder_optimizer = enc_optimizer
        self.generator_optimizer = gen_optimizer
        self.discriminator_optimizer = disc_optimizer
        self.name = name

    def build(self, var_list):
        self.encoder_optimizer.build(var_list)
        self.generator_optimizer.build(var_list)
        self.discriminator_optimizer.build(var_list)

        super().build(var_list)

    def update_step(self, gradient, variable):
        encoder_grads_and_vars = []
        generator_grads_and_vars = []
        discriminator_grads_and_vars = []

        for grad, var in zip(gradient, variable):
            if 'autoencoder' in var.name:
                encoder_grads_and_vars.append((grad, var))
            elif 'encoder' in var.name:
                generator_grads_and_vars.append((grad, var))
            elif 'discriminator' in var.name:
                discriminator_grads_and_vars.append((grad, var))

        self.encoder_optimizer.apply_gradients(encoder_grads_and_vars)
        self.generator_optimizer.apply_gradients(generator_grads_and_vars)
        self.discriminator_optimizer.apply_gradients(discriminator_grads_and_vars)

    def get_config(self):
        config = super(AAEOptimizer, self).get_config()
        config.update({
            "encoder_optimizer": self.encoder_optimizer,
            "generator_optimizer": self.generator_optimizer,
            "discriminator_optimizer": self.discriminator_optimizer,
        })
        return config


class AAE(keras.Model):
    def __init__(
            self,
            encoder,
            decoder,
            discriminator,
            z_dim,
    ):
        super(AAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.accuracy = tf.keras.metrics.BinaryAccuracy()
        self.autoencoder_loss_fn = None
        self.discriminator_loss_fn = None
        self.generator_loss_fn = None
        self.optimizer = None

    def compile(
        self,
        optimizer=AAEOptimizer(),
        generator_loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        autoencoder_loss_fn=tf.keras.losses.MeanSquaredError(),
        discriminator_loss_fn=discriminator_loss,
        **kwargs
    ):
        super(AAE, self).compile(**kwargs)
        self.optimizer = optimizer
        self.generator_loss_fn = generator_loss_fn
        self.autoencoder_loss_fn = autoencoder_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn

    def train_step(self, batch_data):
        batch_x, batch_y = batch_data
        with tf.GradientTape() as ae_tape, tf.GradientTape() as dc_tape, tf.GradientTape() as gen_tape:
            # -------------------------------------------------------------------------------------------------------------
            # Autoencoder
            encoder_output = self.encoder(batch_x, training=True)
            decoder_output = self.decoder(encoder_output, training=True)

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
            gen_loss = self.generator_loss_fn(tf.ones_like(dc_fake), dc_fake)

        ae_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        ae_vars.name = 'autoencoder'
        ae_grads = ae_tape.gradient(ae_loss, ae_vars)
        dc_grads = dc_tape.gradient(dc_loss, self.discriminator.trainable_variables)
        gen_grads = gen_tape.gradient(gen_loss, self.encoder.trainable_variables)

        # Merge the gradients from the different tapes
        grads_and_vars = (list(zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables)) +
                          list(zip(dc_grads, self.discriminator.trainable_variables)) +
                          list(zip(gen_grads, self.encoder.trainable_variables)))

        # Apply the gradients using the custom optimizer
        self.optimizer.apply_gradients(grads_and_vars)

        return {
            "ae_loss": ae_loss,
            "dc_loss": dc_loss,
            "dc_acc": dc_acc,
            "gen_loss": gen_loss,
        }

    def eval(self, data):
        data = data.batch(len(data))
        batch = next(iter(data))
        output = self(batch)
        x, _ = batch
        results = tf.keras.metrics.MSE(x, output)
        return results

    def call(self, data, training=False, mask=None):
        x, y = data
        encoder_output = self.encoder(x, training=False)
        decoder_output = self.decoder(encoder_output, training=False)
        return decoder_output

    def get_config(self):
        return {
            "encoder_config": self.encoder.get_config(),
            "decoder_config": self.decoder.get_config(),
            "discriminator_config": self.discriminator.get_config(),
            "z_dim": self.z_dim,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        encoder = AAEEncoder.from_config(config["encoder_config"], custom_objects=custom_objects)
        decoder = AAEDecoder.from_config(config["decoder_config"], custom_objects=custom_objects)
        discriminator = AAEDiscriminator.from_config(config["discriminator_config"], custom_objects=custom_objects)
        z_dim = config["z_dim"]
        return cls(encoder, decoder, discriminator, z_dim)

#%%
