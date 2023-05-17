import tensorflow as tf
from models.vae_models import VAE, create_vae_decoder, create_vae_encoder
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import numpy as np
import os
from tqdm import tqdm


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


def train_val_vae(vae, train_data, val_data, early_stop=None, epochs=200, savefile=None, lr_scheduler=None, verbose=1):
    """ Train and validate a VAE model.

    Trains and validates a VAE model on the given training and validation data. The model is trained for the given
    number of epochs and the best model is saved if save=True.

    Args:
        vae (VAE): VAE model to train. Needs to be compiled with the desired optimizers, loss functions and callbacks.
        train_data (tf.data.Dataset): Training data.
        val_data (tf.data.Dataset): Validation data.
        early_stop (int): Number of epochs to wait before early stopping. If None, early stopping is disabled.
        epochs (int): Number of epochs to train for.
        savefile (str): Path to save the best model to.
        lr_scheduler (tf.keras.callbacks.LearningRateScheduler): Learning rate scheduler.
        verbose (int): Verbosity level passed to fit.

    Returns: Best model, as determined by the lowest validation loss.
    """
    if early_stop is not None:
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=early_stop)
    else:
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=epochs)

    if lr_scheduler is None:
        hist = vae.fit(train_data, epochs=epochs, validation_data=val_data,
                       callbacks=[stop_early], verbose=verbose)
    else:
        hist = vae.fit(train_data, epochs=epochs, validation_data=val_data,
                       callbacks=[stop_early, lr_scheduler], verbose=verbose)

    if savefile is not None:
        save_vae(vae, savefile)

    return vae, hist


def get_filename_from_params(params):
    """ Create a filename from the given parameters.

    Args:
        params (dict): Dictionary of parameters.

    Returns: Filename created from the given parameters.
    """
    filename = 'vae_'

    enc_hidden_dim = params['encoder'].get('hidden_dim')
    enc_latent_dim = params['encoder'].get('latent_dim')
    enc_dropout_rate = params['encoder'].get('dropout_rate')
    enc_initializer = params['encoder'].get('initializer')
    enc_activation = params['encoder'].get('activation')

    if enc_hidden_dim is not None:
        filename += 'h_dim_'
        for dim in enc_hidden_dim:
            filename += f'{dim}_'

    if enc_latent_dim is not None:
        filename += f'z_dim_{enc_latent_dim}_'

    if enc_dropout_rate is not None:
        enc_dropout_rate_str = str(enc_dropout_rate).replace('.', '')
        filename += f'dropout_{enc_dropout_rate_str}_'

    if enc_initializer is not None:
        filename += f'init_{enc_initializer}_'

    if enc_activation is not None:
        filename += f'act_{enc_activation}'

    return filename


class VAECrossValidator:
    def __init__(self, param_grid, input_dim, k_folds, batch_size=256, save_path='../outputs/models/vae'):
        self.param_grid = param_grid
        self.save_path = save_path
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.kf = KFold(n_splits=k_folds)

    def cross_validate(self, data, epochs=200, verbose=1):
        results = []
        # Shuffle and batch the data
        data = data.shuffle(10000).batch(self.batch_size)

        for params in self.param_grid:
            filename = get_filename_from_params(params)
            savefile = os.path.join(self.save_path, filename)
            val_total_losses = []
            val_recon_losses = []
            training_losses = []
            val_kl_losses = []
            kl_losses = []
            metrics = {}
            vae = None
            print(f"\nTraining model with parameters {params}")
            for i in tqdm(range(self.kf.n_splits), desc="Fold Progress", ncols=80):

                # Load model if it exists, otherwise create a new one
                # TODO: Save losses and load losses to add to results
                # if os.path.exists(savefile):
                #    if verbose > 0:
                #        print(f"Loading model from {savefile}")
                #    vae = load_vae(savefile)

                # else:
                if verbose > 0:
                    print(f"Creating new model for parameters {params}")
                encoder = create_vae_encoder(input_dim=self.input_dim, **params['encoder'])
                decoder = create_vae_decoder(output_dim=self.input_dim, **params['decoder'])
                vae = VAE(encoder, decoder, **params['vae'])
                vae.compile()

                # Create train and validation datasets for this fold
                val_data = data.shard(self.kf.n_splits, i)
                train_data = data.shard(self.kf.n_splits, (i+1) % self.kf.n_splits)

                for j in range(2, self.kf.n_splits):
                    train_data = train_data.concatenate(data.shard(self.kf.n_splits, (i+j) % self.kf.n_splits))

                vae, hist = train_val_vae(vae, train_data, val_data, verbose=verbose, epochs=epochs)
                val_total_losses.append(np.min(hist.history['val_total_loss']))
                val_recon_losses.append(np.min(hist.history['val_reconstruction_loss']))
                val_kl_losses.append(np.min(hist.history['val_kl_loss']))
                training_losses.append((hist.history['total_loss']))
                kl_losses.append(hist.history['kl_loss'])

            if not os.path.exists(savefile):
                save_vae(vae, savefile)
            metrics['total_loss'] = np.mean(val_total_losses)
            metrics['recon_loss'] = np.mean(val_recon_losses)
            metrics['kl_loss'] = np.mean(val_kl_losses)
            metrics['avg_training_losses'] = np.mean(training_losses, axis=0)
            metrics['avg_kl_losses'] = np.mean(kl_losses, axis=0)
            results.append((params, metrics))
        return results


def create_param_grid(h_dims, z_dims, dropouts, activations, initializers, betas=None):
    # Define your base parameter dictionary
    param_dict = {
        'hidden_dim': h_dims,
        'latent_dim': z_dims,
        'dropout_rate': dropouts,
        'activation': activations,
        'initializer': initializers,
    }

    param_dict_vae = {}
    if betas is not None:
        param_dict_vae['beta'] = betas

    # Create the parameter grid
    param_grid = ParameterGrid(param_dict)

    param_grid_vae = ParameterGrid(param_dict_vae)

    # Convert the grid into the format needed for your cross_validate method
    formatted_param_grid = [
        {
            'encoder': {key: params[key] for key in params},
            'decoder': {key: params[key] for key in params},
            'vae': {key: params_vae[key] for key in params_vae}
        }
        for params in param_grid for params_vae in param_grid_vae
    ]
    return formatted_param_grid

#%%
