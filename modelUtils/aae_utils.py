import tensorflow as tf
import numpy as np
from models.aae_models import AAE, AAEEncoder, AAEDecoder, AAEDiscriminator, \
    discriminator_loss
import os
import math


def train_aae(data, batch_size=256, epochs=200, lr=0.0001, h_dim=None, z_dim=20, savefile=None):
    """Create, train and save an AAE model.

    Args:
        data (tf.data.Dataset):
         Dataset to train the model on.
        batch_size (int, optional):
         Batch size to use for training.
        epochs (int, optional):
         Number of epochs to train for.
        lr (float, optional):
         Learning rate to use for training.
        h_dim (list, optional):
         List of hidden layer dimensions.
        z_dim (int, optional):
         Dimension of latent space.
        savefile (str, optional):
         Path to save the model to.

    Returns:
        None
    """
    if h_dim is None:
        h_dim = [100, 100]

    # Batch data
    data = data.batch(batch_size)
    n_features = data.element_spec[0].shape[1]
    n_labels = data.element_spec[1].shape[1]
    aae = create_aae(n_features, n_labels, h_dim, z_dim)

    step_size = 2 * np.ceil(4600 / batch_size)
    # Create learning rate schedules
    encoder_lr_schedule = CyclicLR(lr, step_size, max_lr=0.001)
    generator_lr_schedule = CyclicLR(lr, step_size, max_lr=0.001)
    discriminator_lr_schedule = CyclicLR(lr, step_size, max_lr=0.0005)

    # Create optimizers
    encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile model
    aae.compile(encoder_optimizer, generator_optimizer, discriminator_optimizer,
                discriminator_loss_fn=discriminator_loss)

    # Create learning rate scheduler callback
    lr_schedules = [encoder_lr_schedule, discriminator_lr_schedule, generator_lr_schedule]
    optimizers = [aae.autoencoder_optimizer, aae.discriminator_optimizer, aae.generator_optimizer]
    lr_callback = MultiOptimizerLearningRateScheduler(lr_schedules, optimizers)
    callbacks = [lr_callback]

    # Train model
    aae.fit(data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    # Save model
    if savefile is not None:
        if not os.path.exists(savefile):
            os.makedirs(savefile)
        aae.encoder.save(os.path.join(savefile, 'encoder'), save_format="tf")
        aae.decoder.save(os.path.join(savefile, 'decoder'), save_format="tf")
        aae.discriminator.save(os.path.join(savefile, 'discriminator'), save_format="tf")


def test_aae(data, savefile):
    """Evaluate an AAE model over a dataset.

    Args:
        data (tf.data.Dataset):
            Dataset to evaluate the model on.
        savefile (str):
            Path to the model to evaluate.

    Returns:
        Output of the model on the dataset. Stored as a list of loss values for each sample.
    """

    # restore encoder, decoder, discriminator from savefile
    encoder = tf.keras.models.load_model(os.path.join(savefile, 'encoder'))
    decoder = tf.keras.models.load_model(os.path.join(savefile, 'decoder'))
    discriminator = tf.keras.models.load_model(os.path.join(savefile, 'discriminator'))
    z_dim = encoder.layers[1].output_shape[1]

    # create aae instance
    aae = AAE(encoder, decoder, discriminator, z_dim)
    aae.compile()

    # run test data through aae, saving resulting loss values for each sample
    output = aae.eval(data)
    return output


def create_aae(n_features, n_labels, h_dim, z_dim):
    encoder = AAEEncoder(n_features, h_dim, z_dim)
    decoder = AAEDecoder(z_dim + n_labels, n_features, h_dim)
    discriminator = AAEDiscriminator(z_dim, h_dim)
    aae = AAE(encoder, decoder, discriminator, z_dim)
    return aae


class MultiOptimizerLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Learning rate scheduler that can be used with multiple optimizers."""
    def __init__(self, lr_schedules, optimizers, **kwargs):
        super(MultiOptimizerLearningRateScheduler, self).__init__(lr_schedules[0], **kwargs)
        self.lr_schedules = lr_schedules
        self.optimizers = optimizers

    def on_epoch_begin(self, epoch, logs=None):
        for i, optimizer in enumerate(self.optimizers):
            lr = self.lr_schedules[i].step()
            optimizer.lr.assign(lr)


class CyclicLR:
    """Scheduler that implements a cyclical learning rate policy (CLR).

     Cycles the learning rate between two boundaries with some constant frequency,
     as detailed in the paper `Cyclical Learning Rates for Training Neural Networks`_.
     """
    def __init__(self, base_lr, step_size, max_lr=.005, mode="triangular"):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.iteration = 0

    def get_lr(self):
        cycle = math.floor(1 + self.iteration / (2 * self.step_size))
        x = abs(self.iteration / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        elif self.mode == "triangular2":
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / (2 ** (cycle - 1))
        elif self.mode == "exp_range":
            gamma = 0.999
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * (gamma ** self.iteration)
        else:
            raise ValueError("Invalid mode: choose from 'triangular', 'triangular2', or 'exp_range'.")

        return lr

    def step(self):
        lr = self.get_lr()
        self.iteration += 1
        return lr
#%%
