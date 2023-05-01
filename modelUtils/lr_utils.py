import math
import tensorflow as tf
from keras.callbacks import Callback, LambdaCallback
import numpy as np


class ExponentialDecayScheduler(Callback):
    def __init__(self, initial_lr, decay_rate, decay_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def on_batch_end(self, batch, logs=None):
        new_lr = self.initial_lr * self.decay_rate**(batch / self.decay_steps)
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)


class CyclicLRScheduler(Callback):
    def __init__(self, base_lr, max_lr, step_size, mode="triangular"):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode

    def on_batch_begin(self, batch, logs=None):
        cycle = math.floor(1 + batch / (2 * self.step_size))
        x = abs(batch / self.step_size - 2 * cycle + 1)
        if self.mode == "triangular":
            scale_factor = 1
        elif self.mode == "triangular2":
            scale_factor = 1 / (2**(cycle - 1))
        elif self.mode == "exp_range":
            scale_factor = self.gamma ** batch
        else:
            raise ValueError("Invalid mode provided: " + self.mode)

        new_lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)


def get_lr_scheduler(scheduler, **kwargs):
    if scheduler == "exponential_decay":
        return ExponentialDecayScheduler(kwargs['initial_lr'], kwargs['decay_rate'], kwargs['decay_steps'])
    elif scheduler == "cyclic":
        return CyclicLRScheduler(kwargs['base_lr'], kwargs['max_lr'], kwargs['step_size'], kwargs['mode'])
    else:
        raise ValueError("Invalid scheduler provided: " + scheduler)


def find_learning_rate(train_data, model, min_lr=1e-8, max_lr=1, n_steps=100, optimizer_name=None):
    """Find the optimal learning rate for an Autoencoder model.

    Args:
        train_data (tf.data.Dataset): Dataset to train the model on.
        model (tf.keras.Model): AE model to test the learning rates on.
        min_lr (float, optional): Minimum learning rate to test. Defaults to 1e-8.
        max_lr (float, optional): Maximum learning rate to test. Defaults to 1.
        n_steps (int, optional): Number of batches to train the model for. Defaults to 100.
        optimizer_name (str, optional): Name of the optimizer to use. Defaults to None, which uses the default optimizer
        of the model.

    Returns:
        A tuple containing the learning rates and corresponding losses.
    """
    # Change batch size to modify the number of iterations for the learning rate finder
    batch_size = 128

    # Batch data
    train_data = train_data.batch(batch_size).repeat()
    n_features = train_data.element_spec[0].shape[1]

    # Create optimizer
    initial_lr = min_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    # Update optimizer of model, add support for models with different optimizers
    if optimizer_name is None:
        model.optimizer = optimizer
    elif optimizer_name == "autoencoder":
        model.autoencoder_optimizer = optimizer
    elif optimizer_name == "discriminator":
        model.discriminator_optimizer = optimizer
    elif optimizer_name == "generator":
        model.generator_optimizer = optimizer

    # Define learning rate update function
    def update_learning_rate(batch, logs):
        nonlocal initial_lr
        factor = np.exp(np.log(max_lr / min_lr) / n_steps)
        initial_lr = initial_lr * factor
        tf.keras.backend.set_value(optimizer.lr, initial_lr)

    # Define learning rate update callback
    lr_update_callback = LambdaCallback(on_batch_end=update_learning_rate)

    # Store losses and learning rates
    rec_losses = []
    kl_losses = []
    lrs = []

    def save_losses(batch, logs):
        rec_losses.append(logs['reconstruction_loss'])
        kl_losses.append(logs['kl_loss'])
        lrs.append(tf.keras.backend.get_value(optimizer.lr))

    # Define loss saving callback
    loss_saving_callback = LambdaCallback(on_batch_end=save_losses)

    # Train model with learning rate finder
    model.fit(train_data, callbacks=[lr_update_callback, loss_saving_callback], steps_per_epoch=n_steps, epochs=1,
              verbose=0)

    return lrs, rec_losses, kl_losses