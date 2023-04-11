import tensorflow as tf
import numpy as np
from models.vae_models import VAE, VAEEncoder, VAEDecoder
import os
import math

# TODO: Add exponential decaying learning rate callback


def train_vae(data, batch_size=256, epochs=200, lr=0.0001, h_dim=None, z_dim=20, l2_reg=0.001, savefile=None):
    """Create, train, and save a VAE model

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
        l2_reg (float, optional):
         L2 regularization parameter used to regularize layers in the encoder and decoder.
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

    # Create vae model
    vae = create_vae(n_features, h_dim, z_dim, n_labels, l2_reg=l2_reg)

    # Create optimizer
    vae_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile model
    vae.compile(vae_optimizer)

    # Train model
    vae.fit(data, batch_size=batch_size, epochs=epochs)

    # Save model
    if savefile is not None:
        if not os.path.exists(savefile):
            os.makedirs(savefile)
        vae.encoder.save(os.path.join(savefile, 'encoder'), save_format="tf")
        vae.decoder.save(os.path.join(savefile, 'decoder'), save_format="tf")


def test_vae(data, savefile):
    """Evaluate a VAE model over a dataset.

    Args:
        data (tf.data.Dataset):
            Dataset to evaluate the model on.
        savefile (str):
            Path to the model to evaluate.

    Returns:
        Output of the model on the dataset. Stored as a list of loss values for each sample.
    """

    # restore encoder, decoder from savefile
    encoder = tf.keras.models.load_model(os.path.join(savefile, 'encoder'))
    decoder = tf.keras.models.load_model(os.path.join(savefile, 'decoder'))

    # create vae instance
    vae = VAE(encoder, decoder)
    vae.compile()

    # run test data through vae, saving resulting loss values for each sample
    output = vae.eval(data)
    return output


def create_vae(n_features, h_dim, z_dim, n_labels, l2_reg=1e-3):
    encoder = VAEEncoder(h_dim, z_dim,n_labels, l2_reg=l2_reg)
    decoder = VAEDecoder(h_dim, n_features, l2_reg=l2_reg)
    vae = VAE(encoder, decoder)
    return vae
