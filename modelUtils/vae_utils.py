import pandas as pd
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

    VALIDATION DATA AND TRAINING DATA MUST BE BATCHED

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

    Example filename for a VAE with 2 hidden layers of [100, 100], 10 latent dimensions and beta=0.1 trained for 200
    epochs: 'h_100100z_10b_1e_200'

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
        enc_beta_str = str(enc_beta).replace('.', '')
        filename += f'b_{enc_beta_str}'

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
            r2 = []
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
                    r2.append(np.max(hist.history['val_r2']))
                    val_total.append(hist.history['val_total_loss'])
                    val_recon.append(hist.history['val_reconstruction_loss'])
                    val_kl.append(hist.history['val_kl_loss'])
                    training_losses.append((hist.history['total_loss']))
                    kl_losses.append(hist.history['kl_loss'])

                metrics['Total Loss'] = np.mean(val_total_losses)
                metrics['Reconstruction Loss'] = np.mean(val_recon_losses)
                metrics['KL Loss'] = np.mean(val_kl_losses)
                metrics['R2'] = np.mean(r2)
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

    def cross_validate_df(self, data, epochs=200, verbose=1):
        results = pd.DataFrame(columns=['Hidden Dimensions', 'Latent Dimensions', 'Beta', 'Dropout', 'Activation',
                                        'Initializer', 'Total Loss', 'Reconstruction Loss', 'KL Loss', 'R2',
                                        'Validation Total Loss History', 'Validation Reconstruction Loss History',
                                        'Validation KL Loss History', 'Training Total Loss History',
                                        'Training KL Loss History', 'Training Reconstruction Loss History',
                                        'Parameters'])
        data = data.shuffle(10000).batch(self.batch_size)
        for params in self.param_grid:
            new_row = {}
            filename = get_filename_from_params(params, epochs)
            save_dir = os.path.join(self.save_path, filename)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            val_total_losses = []
            val_recon_losses = []
            val_kl_losses = []
            training_recon_losses = []
            training_kl_losses = []
            training_total_losses = []
            kl_losses = []
            val_total = []
            val_recon = []
            val_kl = []
            r2 = []
            savefile = os.path.join(save_dir, 'results.pkl')
            if os.path.exists(savefile):
                print(f"Loading model results from {savefile}")
                with open(savefile, 'rb') as f:
                    new_row = pickle.load(f)
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
                    r2.append(np.max(hist.history['val_r2']))
                    val_total.append(hist.history['val_total_loss'])
                    val_recon.append(hist.history['val_reconstruction_loss'])
                    val_kl.append(hist.history['val_kl_loss'])
                    training_recon_losses.append((hist.history['reconstruction_loss']))
                    training_kl_losses.append(hist.history['kl_loss'])
                    training_total_losses.append(hist.history['total_loss'])

                    new_row['Total Loss'] = np.mean(val_total_losses)
                    new_row['Reconstruction Loss'] = np.mean(val_recon_losses)
                    new_row['KL Loss'] = np.mean(val_kl_losses)
                    new_row['R2'] = np.mean(r2)
                    new_row['Validation Total Loss History'] = np.mean(val_total, axis=0)
                    new_row['Validation Reconstruction Loss History'] = np.mean(val_recon, axis=0)
                    new_row['Validation KL Loss History'] = np.mean(val_kl, axis=0)
                    new_row['Training Total Loss History'] = np.mean(training_total_losses, axis=0)
                    new_row['Training Reconstruction Loss History'] = np.mean(training_recon_losses, axis=0)
                    new_row['Training KL Loss History'] = np.mean(training_kl_losses, axis=0)
                    new_row['KL Loss History'] = np.mean(kl_losses, axis=0)
                    new_row['Parameters'] = params
                    new_row['Hidden Dimensions'] = params['encoder']['hidden_dim']
                    new_row['Latent Dimensions'] = params['encoder']['latent_dim']
                    new_row['Beta'] = params['vae']['beta']
                    new_row['Dropout'] = params['encoder']['dropout_rate']
                    new_row['Activation'] = params['encoder']['activation']
                    new_row['Initializer'] = params['encoder']['initializer']

                    with open(savefile, 'wb') as f:
                        pickle.dump(new_row, f)
                    save_vae(vae, save_dir)
            results.loc[len(results)] = new_row
        return results

    def cross_validate_df_val(self, train_data, val_data, epochs=200, verbose=1):
        results = pd.DataFrame(columns=['Hidden Dimensions', 'Latent Dimensions', 'Beta', 'Dropout', 'Activation',
                                        'Initializer', 'Total Loss', 'Reconstruction Loss', 'KL Loss', 'R2',
                                        'Validation Total Loss History', 'Validation Reconstruction Loss History',
                                        'Validation KL Loss History', 'Training Total Loss History',
                                        'Training KL Loss History', 'Training Reconstruction Loss History',
                                        'Parameters'])
        data = train_data.shuffle(10000).batch(self.batch_size)
        val_data = val_data.batch(val_data.cardinality().numpy())
        for params in self.param_grid:
            new_row = {}
            filename = get_filename_from_params(params, epochs)
            save_dir = os.path.join(self.save_path, filename)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            val_total_losses = []
            val_recon_losses = []
            val_kl_losses = []
            training_recon_losses = []
            training_kl_losses = []
            training_total_losses = []
            kl_losses = []
            val_total = []
            val_recon = []
            val_kl = []
            r2_list = []
            savefile = os.path.join(save_dir, 'results.pkl')
            if os.path.exists(savefile):
                print(f"Loading model results from {savefile}")
                with open(savefile, 'rb') as f:
                    new_row = pickle.load(f)
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
                    cv_val_data = data.shard(self.kf.n_splits, i)
                    train_data = data.shard(self.kf.n_splits, (i+1) % self.kf.n_splits)

                    for j in range(2, self.kf.n_splits):
                        train_data = train_data.concatenate(data.shard(self.kf.n_splits, (i+j) % self.kf.n_splits))

                    vae, hist = train_val_vae(vae, train_data, cv_val_data, verbose=verbose, epochs=epochs)

                    val_total_losses.append(np.min(hist.history['val_total_loss']))
                    val_recon_losses.append(np.min(hist.history['val_reconstruction_loss']))
                    val_kl_losses.append(np.min(hist.history['val_kl_loss']))
                    val_total.append(hist.history['val_total_loss'])
                    val_recon.append(hist.history['val_reconstruction_loss'])
                    val_kl.append(hist.history['val_kl_loss'])
                    training_recon_losses.append((hist.history['reconstruction_loss']))
                    training_kl_losses.append(hist.history['kl_loss'])
                    training_total_losses.append(hist.history['total_loss'])
                    _, r2, _, _ = vae.evaluate(val_data, verbose=verbose)
                    r2_list.append(r2)

                new_row['Total Loss'] = np.mean(val_total_losses)
                new_row['Reconstruction Loss'] = np.mean(val_recon_losses)
                new_row['KL Loss'] = np.mean(val_kl_losses)
                new_row['Validation Total Loss History'] = np.mean(val_total, axis=0)
                new_row['Validation Reconstruction Loss History'] = np.mean(val_recon, axis=0)
                new_row['Validation KL Loss History'] = np.mean(val_kl, axis=0)
                new_row['Training Total Loss History'] = np.mean(training_total_losses, axis=0)
                new_row['Training Reconstruction Loss History'] = np.mean(training_recon_losses, axis=0)
                new_row['Training KL Loss History'] = np.mean(training_kl_losses, axis=0)
                new_row['KL Loss History'] = np.mean(kl_losses, axis=0)
                new_row['R2'] = np.mean(r2_list)
                new_row['Parameters'] = params
                new_row['Hidden Dimensions'] = params['encoder']['hidden_dim']
                new_row['Latent Dimensions'] = params['encoder']['latent_dim']
                new_row['Beta'] = params['vae']['beta']
                new_row['Dropout'] = params['encoder']['dropout_rate']
                new_row['Activation'] = params['encoder']['activation']
                new_row['Initializer'] = params['encoder']['initializer']

                with open(savefile, 'wb') as f:
                    pickle.dump(new_row, f)
                save_vae(vae, save_dir)
            results.loc[len(results)] = new_row
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
