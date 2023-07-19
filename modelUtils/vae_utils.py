import tensorflow as tf
from models.vae_models import VAE, create_vae_decoder, create_vae_encoder
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import numpy as np
import os
from tqdm import tqdm
import pickle


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


def create_vae(n_features, h_dim, z_dim, beta=1.0):
    """ Creates a VAE model with the given parameters.

    Creates an un-compiled VAE model with the given parameters. The model must be compiled with the desired
    optimizers, loss functions and callbacks before it can be trained.

    Args:
        n_features (int): Number of features in the input data.
        h_dim (list): List of hidden layer dimensions.
        z_dim (int): Dimension of latent space.
        beta (float): Beta parameter for the KL divergence loss.

    Returns: Un-compiled VAE model with the given parameters.
    """
    encoder = create_vae_encoder(n_features, h_dim, z_dim)
    decoder = create_vae_decoder(z_dim, h_dim, n_features)
    vae = VAE(encoder, decoder, beta=beta)
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

    Returns: Best trained VAE model if early stopping is enabled, otherwise the model trained for the given number of
    epochs. Also returns the training history.
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


def get_filename_from_params(params, epochs):
    """ Create a filename from the given parameters.

    Args:
        params (dict): Dictionary of parameters.
        epochs (int): Number of epochs to train for.

    Returns: Filename created from the given parameters.
    """
    filename = ''

    enc_hidden_dim = params['encoder'].get('hidden_dim')
    enc_latent_dim = params['encoder'].get('latent_dim')
    enc_beta = params['vae'].get('beta')

    if enc_hidden_dim is not None:
        filename += 'h_'
        for dim in enc_hidden_dim:
            filename += f'{dim}'

    if enc_latent_dim is not None:
        filename += f'z_{enc_latent_dim}'

    if enc_beta is not None:
        filename += f'b_{enc_beta}'

    filename += f'e_{epochs}'
    return filename


def load_or_train_model(path, params, train_data, epochs):
    """ Load a VAE model from the given filepath.

    If a model exists at the given filepath, it is loaded and returned, otherwise a new model is created and returned.

    Args:
        path (str): Path to load the model from.
        params (dict): Dictionary of parameters.
        epochs (int): Desired training epochs.
        train_data (tf.data.Dataset): Training data.

    Returns: VAE model loaded from the given filepath or a new model.
    """

    filepath = os.path.join(path, get_filename_from_params(params, epochs))
    input_dim = train_data.element_spec[0].shape[0]
    if os.path.exists(filepath):
        vae = load_vae(filepath)
        vae.compile()
        print(f'Loaded model from {filepath}.')
    else:
        print(f'No model found at {filepath}. Creating new model.')
        encoder = create_vae_encoder(input_dim=input_dim, **params['encoder'])
        decoder = create_vae_decoder(output_dim=input_dim, **params['decoder'])
        vae = VAE(encoder, decoder, **params['vae'])
        vae.compile()
        print('Training model.')
        vae.fit(train_data.batch(128), epochs=epochs, verbose=0)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        save_vae(vae, filepath)
        print(f'Saved model to {filepath}.')
    return vae


class VAECrossValidator:
    def __init__(self, param_grid, input_dim, k_folds, save_path, batch_size=256):
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
            filename = get_filename_from_params(params, epochs)
            save_dir = os.path.join(self.save_path, filename)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            val_total_losses = []
            val_recon_losses = []
            training_losses = []
            val_kl_losses = []
            kl_losses = []
            val_total = []
            val_recon = []
            val_kl = []
            metrics = {}
            vae = None
            savefile = os.path.join(save_dir, 'results.pkl')
            if os.path.exists(savefile):
                print(f"Loading model results from {savefile}")
                with open(savefile, 'rb') as f:
                    metrics = pickle.load(f)
            else:
                print(f"\nTraining model with parameters {params}")
                for i in tqdm(range(self.kf.n_splits), desc="Fold Progress", ncols=80):
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
                    val_total.append(hist.history['val_total_loss'])
                    val_recon.append(hist.history['val_reconstruction_loss'])
                    val_kl.append(hist.history['val_kl_loss'])
                    training_losses.append((hist.history['total_loss']))
                    kl_losses.append(hist.history['kl_loss'])
                metrics['total_loss'] = np.mean(val_total_losses)
                metrics['recon_loss'] = np.mean(val_recon_losses)
                metrics['kl_loss'] = np.mean(val_kl_losses)
                metrics['avg_val_total_losses'] = np.mean(val_total, axis=0)
                metrics['avg_val_recon_losses'] = np.mean(val_recon, axis=0)
                metrics['avg_val_kl_losses'] = np.mean(val_kl, axis=0)
                metrics['avg_training_losses'] = np.mean(training_losses, axis=0)
                metrics['avg_kl_losses'] = np.mean(kl_losses, axis=0)
                with open(savefile, 'wb') as f:
                    pickle.dump(metrics, f)
                save_vae(vae, save_dir)
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
