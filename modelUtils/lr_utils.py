import math
import tensorflow as tf
from keras.callbacks import LambdaCallback
import numpy as np


class ExponentialDecayScheduler:
    """Scheduler that implements exponential decay.

    Implements exponential decay of the learning rate with a given decay rate and decay steps. The learning rate
    is updated via callbacks at the beginning of each epoch. The learning rate is updated according to the formula:
    lr = initial_lr * decay_rate^(iteration / decay_steps)

    Args:
        initial_lr (float): Initial learning rate.
        decay_rate (float): Decay rate.
        decay_steps (int): Number of steps between each decay.

    Attributes:
        initial_lr (float): Initial learning rate.
        decay_rate (float): Decay rate.
        decay_steps (int): Number of steps between each decay.
        lr (float): Current learning rate.
        iteration (int): Current iteration.
    """

    def __init__(self, initial_lr, decay_rate, decay_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.lr = initial_lr
        self.iteration = 0

    def get_lr(self):
        new_lr = self.initial_lr * self.decay_rate**(self.iteration / self.decay_steps)
        return new_lr

    def step(self):
        self.lr = self.get_lr()
        self.iteration += 1
        return self.lr


class CyclicLR:
    """Scheduler that implements a cyclical learning rate policy (CLR).

     Cycles the learning rate between two boundaries with some constant frequency,
     as detailed in the paper `Cyclical Learning Rates for Training Neural Networks`_.

     Args:
        base_lr (float): Initial learning rate which is the lower boundary in the cycle.
        max_lr (float): Upper boundary in the cycle. Functionally, it defines the cycle amplitude (max_lr - base_lr).
        step_size (int): Number of training iterations per half cycle. Authors suggest setting step_size equal to 2-8
        x training iterations in epoch.

    Attributes:
        base_lr (float): Initial learning rate which is the lower boundary in the cycle.
        max_lr (float): Upper boundary in the cycle. Functionally, it defines the cycle amplitude (max_lr - base_lr).
        lr (float): Current learning rate.
        step_size (int): Number of training iterations per half cycle. Authors suggest setting step_size
        mode (str): One of {triangular, triangular2, exp_range}. Default 'triangular'.
        iteration (int): Current iteration.
     """
    def __init__(self, base_lr, step_size, max_lr=.005, mode="triangular"):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.lr = base_lr
        self.step_size = step_size
        self.mode = mode
        self.iteration = 0

    def get_lr(self):
        cycle = math.floor(1 + self.iteration / (2 * self.step_size))
        x = abs(self.iteration / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0.0, (1 - x))
        elif self.mode == "triangular2":
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0.0, (1 - x)) / (2 ** (cycle - 1))
        elif self.mode == "exp_range":
            gamma = 0.999
            lr = self.base_lr + (self.max_lr - self.base_lr) * max(0.0, (1 - x)) * (gamma ** self.iteration)
        else:
            raise ValueError("Invalid mode: choose from 'triangular', 'triangular2', or 'exp_range'.")
        return lr

    def step(self):
        self.lr = self.get_lr()
        self.iteration += 1
        return self.lr


class MultiOptimizerLearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    """Learning rate scheduler that can be used with multiple optimizers.

    Assigns learning rates to each of the optimizer, learning rate pairs provided. The learning rate
    is updated via callbacks at the beginning of each epoch.
    Args:
        lr_and_optimizers (list): List of tuples of the form (lr_schedule, optimizer), with each lr_schedule
        corresponding to an optimizer.
    """
    def __init__(self, lr_and_optimizers, **kwargs):
        super(MultiOptimizerLearningRateScheduler, self).__init__(schedule=None, **kwargs)
        self.lr_schedulers = []
        self.optimizers = []
        for lr_schedule, optimizer in lr_and_optimizers:
            self.lr_schedulers.append(lr_schedule)
            self.optimizers.append(optimizer)

    def step(self):
        for i in range(len(self.lr_schedulers)):
            tf.keras.backend.set_value(self.optimizers[i].lr, self.lr_schedulers[i].step())

    def on_epoch_begin(self, epoch, logs=None):
        for i in range(len(self.lr_schedulers)):
            tf.keras.backend.set_value(self.optimizers[i].lr, self.lr_schedulers[i].step())


def find_learning_rate(train_data, model, min_lr=1e-8, max_lr=0.1, n_steps=100, batch_size=128, optimizer_name=None):
    """Find the optimal learning rate for an Autoencoder model.

    Args:
        train_data (tf.data.Dataset): Dataset to train the model on.
        model (tf.keras.Model): AE model to test the learning rates on.
        min_lr (float, optional): Minimum learning rate to test. Defaults to 1e-8.
        max_lr (float, optional): Maximum learning rate to test. Defaults to 1.
        n_steps (int, optional): Number of lr updates to perform between the min and max lr. Defaults to 100.
        batch_size (int, optional): Batch size to use. Defaults to 128.
        optimizer_name (str, optional): Name of the optimizer to use. Defaults to None, which uses the default optimizer
        of the model.

    Returns:
        A tuple containing the learning rates and corresponding losses.
    """

    # Batch data
    train_data = train_data.batch(batch_size).repeat()

    # Create optimizer
    initial_lr = min_lr
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile()
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

    # Store model logs in a dictionary
    log_dict = {}

    def save_losses(batch, logs):
        for key, value in logs.items():
            if key not in log_dict:
                log_dict[key] = []
            log_dict[key].append(value)

    # Define loss saving callback
    loss_saving_callback = LambdaCallback(on_batch_end=save_losses)

    # Train model with learning rate finder
    model.fit(train_data, callbacks=[lr_update_callback, loss_saving_callback], steps_per_epoch=n_steps, epochs=1,
              verbose=0)

    return log_dict
