import tensorflow as tf
from tensorflow import keras


class AAEEncoder(tf.keras.Model):
    """Encoder for the Adversarial Autoencoder

    Creates a custom encoder that takes in an input and outputs a latent representation of the input.

    Args:
        hidden_dim (list): List of hidden dimensions for the encoder
        latent_dim (int): Dimensionality of the latent representation

    Attributes:
        hidden_dim (list): List of hidden dimensions for the encoder
        latent_dim (int): Dimensionality of the latent representation
        hidden_layers (list): List of hidden layers for the encoder
        relu (tf.keras.layers.LeakyReLU): LeakyReLU activation function. Activation function for the hidden layers
        latent_layer (tf.keras.layers.Dense): Dense layer for the latent representation
    """
    def __init__(self, hidden_dim, latent_dim):
        super(AAEEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hidden_layers = []
        for h_dim in self.hidden_dim:
            self.hidden_layers.append(keras.layers.Dense(h_dim))
        self.relu = keras.layers.LeakyReLU()
        # Define the latent representation layer
        self.latent_layer = keras.layers.Dense(self.latent_dim, activation='relu')

    def call(self, x, training=False, mask=None):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        latent = self.latent_layer(x)
        return latent

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["hidden_dim"], config["latent_dim"])


class AAEDecoder(tf.keras.Model):
    """Decoder for the Adversarial Autoencoder

    Creates a custom decoder that takes in a latent representation and outputs a reconstructed representation of the
    input. The decoder is symmetric to the encoder.

    Args:
        input_dim (int): Dimensionality of the input
        hidden_dim (list): List of hidden dimensions for the decoder

    Attributes:
        input_dim (int): Dimensionality of the input
        hidden_dim (list): List of hidden dimensions for the decoder
        hidden_layers (list): List of hidden layers for the decoder
        relu (tf.keras.layers.LeakyReLU): LeakyReLU activation function. Activation function for the hidden layers
        output_layer (tf.keras.layers.Dense): Dense layer for the reconstructed representation
    """
    def __init__(self, input_dim, hidden_dim):
        super(AAEDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.hidden_layers = []
        for h_dim in self.hidden_dim:
            self.hidden_layers.append(keras.layers.Dense(h_dim))
        self.relu = keras.layers.LeakyReLU()
        # Define the output representation layer
        self.output_layer = keras.layers.Dense(self.input_dim, activation='linear')

    def call(self, x, training=False, mask=None):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        output = self.output_layer(x)
        return output

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "input_dim": self.input_dim,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["hidden_dim"], config["input_dim"])


class AAEDiscriminator(tf.keras.Model):
    """Discriminator for the Adversarial Autoencoder

    Creates a custom discriminator that takes in a latent representation and outputs a probability of the latent
    representation being real or fake. The discriminator is symmetric to the encoder.

    Args:
        latent_dim (int): Dimensionality of the latent representation
        hidden_dim (list): List of hidden dimensions for the discriminator

    Attributes:
        latent_dim (int): Dimensionality of the latent representation
        hidden_dim (list): List of hidden dimensions for the discriminator
        hidden_layers (list): List of hidden layers for the discriminator
        relu (tf.keras.layers.LeakyReLU): LeakyReLU activation function. Activation function for the hidden layers
        output_layer (tf.keras.layers.Dense): Dense layer for the probability of the latent representation being real
            or fake
    """
    def __init__(self, latent_dim, hidden_dim):
        super(AAEDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hidden_layers = []
        for h_dim in self.hidden_dim:
            self.hidden_layers.append(keras.layers.Dense(h_dim))
        self.relu = keras.layers.LeakyReLU()
        # Define the output representation layer
        self.output_layer = keras.layers.Dense(1)

    def call(self, x, training=False, mask=None):
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        output = self.output_layer(x)
        return output

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(config["hidden_dim"], config["latent_dim"])


def discriminator_loss(real_output, fake_output):
    """Loss function for the discriminator

    Calculates the loss of the discriminator using the binary cross entropy loss function.

    Args:
        real_output (tf.Tensor): Output of the discriminator for the real input
        fake_output (tf.Tensor): Output of the discriminator for the fake input

    Returns: The loss of the discriminator, which is the sum of the binary cross entropy loss for the real and fake
        inputs
    """
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_real = cross_entropy(tf.ones_like(real_output), real_output)
    loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return loss_fake + loss_real


class AAEOptimizer(tf.keras.optimizers.Optimizer):
    """Optimizer for the Adversarial Autoencoder

    Creates a custom optimizer that updates the encoder, generator, and discriminator using separate optimizers
    which can be customized using the input parameters.

    Args:
        enc_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the encoder, defaults to Adam
        gen_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator, defaults to Adam
        disc_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the discriminator, defaults to Adam
        name (str): Name of the optimizer

    Attributes:
        autoencoder_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the autoencoder
        generator_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator
        discriminator_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the discriminator
        name (str): Name of the optimizer defaults to AAEOptimizer
    """
    def __init__(
        self,
        aenc_optimizer=None,
        gen_optimizer=None,
        disc_optimizer=None,
        lr=0.001,
        name='AAEOptimizer',
        **kwargs
    ):
        super().__init__(
            name=name,
            **kwargs
        )
        self.autoencoder_optimizer = aenc_optimizer or tf.keras.optimizers.Adam(learning_rate=lr)
        self.generator_optimizer = gen_optimizer or tf.keras.optimizers.Adam(learning_rate=lr)
        self.discriminator_optimizer = disc_optimizer or tf.keras.optimizers.Adam(learning_rate=lr)
        self._learning_rate = lr

    def build(self, var_list):
        super().build(var_list)

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        if name == 'autoencoder':
            self.autoencoder_optimizer.apply_gradients(grads_and_vars, **kwargs)
        elif name == 'generator':
            self.generator_optimizer.apply_gradients(grads_and_vars, **kwargs)
        elif name == 'discriminator':
            self.discriminator_optimizer.apply_gradients(grads_and_vars, **kwargs)
        else:
            raise ValueError(f"Invalid optimizer name {name}")

    def update_step(self, gradient, variable):
        pass

    def get_config(self):
        config = super(AAEOptimizer, self).get_config()
        config.update({
            "encoder_optimizer": self.encoder_optimizer,
            "generator_optimizer": self.generator_optimizer,
            "discriminator_optimizer": self.discriminator_optimizer,
        })
        return config


class AAE(keras.Model):
    """Adversarial Autoencoder Model

    Creates a custom adversarial autoencoder model that takes in an encoder, decoder, and discriminator. The encoder
    and decoder are symmetric to each other and the discriminator is symmetric to the encoder and decoder. The
    adversarial autoencoder is trained by minimizing the reconstruction loss and the discriminator loss. The
    reconstruction loss is the mean squared error between the input and the output of the autoencoder. The discriminator
    loss is the binary cross entropy loss between the discriminator output and the true label. The discriminator is
    trained to distinguish between the latent representation of the encoder and the latent representation of the
    generator. The generator is trained to fool the discriminator by generating latent representations that are
    indistinguishable from the encoder's latent representations. The autoencoder is trained to reconstruct the input
    image.

    Args:
        encoder (tf.keras.Model): Encoder model
        decoder (tf.keras.Model): Decoder model
        discriminator (tf.keras.Model): Discriminator model
        z_dim (int): Dimension of the latent representation

    Attributes:
        encoder (tf.keras.Model): Encoder model
        decoder (tf.keras.Model): Decoder model
        discriminator (tf.keras.Model): Discriminator model
        z_dim (int): Dimension of the latent representation
        accuracy (tf.keras.metrics.BinaryAccuracy): Accuracy metric for the discriminator
        autoencoder_loss_fn (tf.keras.losses.Loss): Loss function for the autoencoder
        discriminator_loss_fn (tf.keras.losses.Loss): Loss function for the discriminator
        generator_loss_fn (tf.keras.losses.Loss): Loss function for the generator
        optimizer (AAEOptimizer): Optimizer for the autoencoder, generator, and discriminator
    """
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
        ae_grads = ae_tape.gradient(ae_loss, ae_vars)
        dc_grads = dc_tape.gradient(dc_loss, self.discriminator.trainable_variables)
        gen_grads = gen_tape.gradient(gen_loss, self.encoder.trainable_variables)

        ae_grads_and_vars = zip(ae_grads, ae_vars)
        dc_grads_and_vars = zip(dc_grads, self.discriminator.trainable_variables)
        gen_grads_and_vars = zip(gen_grads, self.encoder.trainable_variables)

        # Apply the gradients using the custom optimizer
        self.optimizer.apply_gradients(ae_grads_and_vars, name='autoencoder')
        self.optimizer.apply_gradients(dc_grads_and_vars, name='discriminator')
        self.optimizer.apply_gradients(gen_grads_and_vars, name='generator')

        return {
            "ae_loss": ae_loss,
            "dc_loss": dc_loss,
            "dc_acc": dc_acc,
            "gen_loss": gen_loss,
            "ae_lr": self.optimizer.autoencoder_optimizer.lr,
            "dc_lr": self.optimizer.discriminator_optimizer.lr,
            "gen_lr": self.optimizer.generator_optimizer.lr,
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
