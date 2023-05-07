import tensorflow as tf
from modelUtils.lr_utils import MultiOptimizerLearningRateScheduler


def train_ae(model, data, batch_size=256, epochs=200, optimizers=None, lr_schedulers=None):
    """Train an autoencoder model

    Compiles an autoencoder with the given optimizers and trains it on the given data.

    Args:
        model (tf.keras.Model): Autoencoder model to train.
        data (tf.data.Dataset): Un-batched Dataset to train the model on.
        batch_size (int, optional): Batch size to use for training.
        epochs (int, optional): Number of epochs to train for.
        optimizers (list, optional): List of optimizers to use for training.
        lr_schedulers (list, optional): List of learning rate schedulers to use for training, needs to be
            the same length as optimizers.

    Returns: The trained autoencoder model.
    """
    if optimizers is None:
        optimizers = [tf.keras.optimizers.Adam(learning_rate=0.0001)]

    # Batch data
    data = data.batch(batch_size)

    # Compile model
    model.compile(optimizers)

    # Create callbacks
    callbacks = []

    # Create learning rate scheduler
    if lr_schedulers is not None:
        lr_callback = MultiOptimizerLearningRateScheduler(lr_schedulers, optimizers)
        callbacks = [lr_callback]

    # Train model
    model.fit(data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    return model


def test_ae(model, data):
    """Test an autoencoder model

    Args:
        data (tf.data.Dataset): Un-batched Dataset to test the model on.

    Returns: The loss of the model on the given data.
    """

    # Batch data
    data = data.batch(256)

    # Test model
    output = model.evaluate(data)

    return output

