import tensorflow as tf
from models.vae_models import VAE, create_vae_decoder, create_vae_encoder
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import numpy as np
import os


def save_vae(vae, savefile):
    """ Save a VAE model.

    Args:
        vae (VAE): VAE model to be saved.
        savefile (str): Path to save the model to.
    """

    if not os.path.exists(savefile):
        os.makedirs(savefile)
    vae.encoder.save(os.path.join(savefile, 'encoder'))
    vae.decoder.save(os.path.join(savefile, 'decoder'))


def load_vae(savefile):
    """ Load a VAE model from a savefile

    Loads and creates a VAE model from a savefile, returns the un-compiled model. If the model needs to be
    trained further it must be compiled with the desired optimizers, loss functions and callbacks.

    Args:
        savefile (str): Path to the model to load.

    Returns: VAE model loaded with the pretrained encoder and decoder weights.
    """

    # restore encoder, decoder from savefile
    encoder = tf.keras.models.load_model(os.path.join(savefile, 'encoder'))
    decoder = tf.keras.models.load_model(os.path.join(savefile, 'decoder'))

    # create vae instance
    vae = VAE(encoder, decoder)
    return vae


def create_vae(n_features, h_dim, z_dim):
    """ Creates a VAE model with the given parameters.

    Creates an un-compiled VAE model with the given parameters. The model must be compiled with the desired
    optimizers, loss functions and callbacks before it can be trained.

    Args:
        n_features (int): Number of features in the input data.
        h_dim (list): List of hidden layer dimensions.
        z_dim (int): Dimension of latent space.

    Returns: Un-compiled VAE model with the given parameters.
    """
    encoder = create_vae_encoder(n_features, h_dim, z_dim)
    decoder = create_vae_decoder(z_dim, h_dim, n_features)
    vae = VAE(encoder, decoder)
    return vae


def train_val_vae(vae, train_data, val_data, batch_size=256, epochs=200, savefile=None, lr_scheduler=None):
    """ Train and validate a VAE model.

    Trains and validates a VAE model on the given training and validation data. The model is trained for the given
    number of epochs and the best model is saved if save=True.

    Args:
        vae (VAE): VAE model to train. Needs to be compiled with the desired optimizers, loss functions and callbacks.
        train_data (tf.data.Dataset): Training data.
        val_data (tf.data.Dataset): Validation data.
        batch_size (int): Batch size.
        epochs (int): Number of epochs to train for.
        savefile (str): Path to save the best model to.
        lr_scheduler (tf.keras.callbacks.LearningRateScheduler): Learning rate scheduler.

    Returns: Best model, as determined by the lowest validation loss.
    """

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=5)

    if lr_scheduler is None:
        hist = vae.fit(train_data.batch(batch_size), epochs=epochs, validation_data=val_data.batch(batch_size),
                       callbacks=[stop_early])
    else:
        hist = vae.fit(train_data.batch(batch_size), epochs=epochs, validation_data=val_data.batch(batch_size),
                       callbacks=[stop_early, lr_scheduler])

    if savefile is not None:
        save_vae(vae, savefile)

    return vae, hist


def get_filename_from_params(params):
    return 'vae_' + str(params['encoder']['hidden_dim']) + '_' + str(params['encoder']['latent_dim']) + '_Encoder_' + \
        str(params['encoder']['dropout_rate']) + '_' + str(params['encoder']['initializer']) + '_' + \
        str(params['encoder']['activation']) + '_Decoder_' + str(params['decoder']['dropout_rate']) + '_' + \
        str(params['decoder']['initializer']) + '_' + str(params['decoder']['activation'])


class VAECrossValidator:
    def __init__(self, param_grid, n_features, k_folds, save_path='outputs/models/vae'):
        self.param_grid = ParameterGrid(param_grid)
        self.n_features = n_features
        self.save_path = save_path
        self.kf = KFold(n_splits=k_folds)

    def cross_validate(self, data):
        results = []
        data = np.array(data)

        for params in self.param_grid:
            filename = get_filename_from_params(params)
            savefile = os.path.join(self.save_path, filename)
            fold_losses = []
            vae = None
            for train_index, val_index in self.kf.split(data):
                if os.path.exists(savefile):
                    print(f"Loading model from {savefile}")
                    vae = load_vae(savefile)
                else:
                    print(f"Creating new model for parameters {params}")
                    encoder = create_vae_encoder(**params['encoder'])
                    decoder = create_vae_decoder(**params['decoder'])
                    vae = VAE(encoder, decoder)
                train_data = data[train_index]
                val_data = data[val_index]
                vae, hist = train_val_vae(vae, train_data, val_data)
                fold_losses.append(hist)
            if not os.path.exists(savefile):
                save_vae(vae, savefile)
            results.append((params, fold_losses))
        return results


#%%
