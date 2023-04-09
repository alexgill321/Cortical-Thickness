import tensorflow as tf
import numpy as np
from models import AAE, Encoder, Decoder, Discriminator, \
    discriminator_loss
import os
import math


def train_aae(data, batch_size=256, epochs=200, lr=0.0001, h_dim=None, z_dim=20, savefile=None):
    """ Create, Compile, and Train AAE model
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
    lr_callback = MultiOptimizerLearningRateScheduler(lr_schedules)
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
    encoder = Encoder(n_features, h_dim, z_dim)
    decoder = Decoder(z_dim + n_labels, n_features, h_dim)
    discriminator = Discriminator(z_dim, h_dim)
    aae = AAE(encoder, decoder, discriminator, z_dim)
    return aae


class MultiOptimizerLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, lr_schedules, **kwargs):
        super(MultiOptimizerLearningRateScheduler, self).__init__(lr_schedules[0], **kwargs)
        self.lr_schedules = lr_schedules

    def on_epoch_begin(self, epoch, logs=None):
        for i, optimizer in enumerate([self.model.autoencoder_optimizer, self.model.discriminator_optimizer,
                                       self.model.generator_optimizer]):
            lr = self.lr_schedules[i].step()
            optimizer.lr.assign(lr)


class CyclicLR:
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
