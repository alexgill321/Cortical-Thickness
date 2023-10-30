import pandas as pd
import tensorflow as tf
from models.vae_models import VAE, create_vae_decoder, create_vae_encoder
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
import numpy as np
import os
from tqdm import tqdm
import pickle
from keras.callbacks import Callback
import itertools
from utils import generate_data
from typing import Any
import inspect


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


def train_val_vae(vae, train_data, val_data, epochs=200, savefile=None, early_stop=None, callbacks=None, verbose=1):

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

    if callbacks is None:
        hist = vae.fit(train_data, epochs=epochs, validation_data=val_data,
                       callbacks=[stop_early], verbose=verbose)
    else:
        callbacks.append(stop_early)
        hist = vae.fit(train_data, epochs=epochs, validation_data=val_data,
                       callbacks=callbacks, verbose=verbose)

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


def get_filename_from_df(row):
    # TODO: implement this
    return "not implemented yet"


def load_or_train_model(params, train_data, epochs, path=None, verbose=0):
    """ Load a VAE model from the given filepath.

    If a model exists at the given filepath, it is loaded and returned, otherwise a new model is created and returned.

    Args:
        path (str): Path to load the model from. Pass 'None' to always create a new model.
        params (dict): Dictionary of parameters.
        epochs (int): Desired training epochs.
        train_data (tf.data.Dataset): Training data.
        verbose (int): Verbosity level passed to fit.

    Returns: VAE model loaded from the given filepath or a new model.
    """
    input_dim = train_data.element_spec[0].shape[0]
    if path is None:
        print("Path is None. Creating a new model.")
        filepath = None
    else:
        filepath = os.path.join(path, get_filename_from_params(params, epochs))

    if filepath and os.path.exists(filepath):
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
        vae.fit(train_data.batch(128), epochs=epochs, verbose=verbose)
        if path is not None:
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
                for i in tqdm(range(self.kf.n_splits), desc="Fold Progress"):
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

    def cross_validate_df_val(self, train_data, val_data, epochs=200, verbose=1, load=False):
        results = pd.DataFrame(columns=['Hidden Dimensions', 'Latent Dimensions', 'Beta', 'Dropout', 'Activation',
                                        'Initializer', 'Total Loss', 'Reconstruction Loss', 'KL Loss', 'R2',
                                        'Validation Total Loss History', 'Validation Reconstruction Loss History',
                                        'Validation KL Loss History', 'Training Total Loss History',
                                        'Training KL Loss History', 'Training Reconstruction Loss History',
                                        'Parameters'])
        data = train_data.shuffle(10000).batch(self.batch_size)
        val_data = val_data.batch(val_data.cardinality().numpy())
        for params in self.param_grid:
            vae = None
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
            if os.path.exists(savefile) and load:
                print(f"Loading model results from {savefile}")
                with open(savefile, 'rb') as f:
                    new_row = pickle.load(f)
            else:
                print(f"\nTraining model with parameters {params}")
                for i in tqdm(range(self.kf.n_splits), desc="Fold Progress"):
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


def create_param_df(conditioning: list[bool] = None, h_dim: list[list[int]] = None, z_dim: list[int] = None,
                    dropout: list[float] = None, activation: list[str] = None, initializer: list[str] = None,
                    beta: list[float] = None, samples: int = None, **kwargs: Any) -> pd.DataFrame:
    """ Creates a parameter grid as a dataframe from the given parameters.

    Args:
        conditioning (list[bool]): List of condition values to use.
        h_dim (list[list[int]]): List of hidden layer dimensions to use.
        z_dim (list[int]): List of latent dimensions to use.
        dropout (list[float]): List of dropout rates to use.
        activation (list[str]): List of activation functions to use.
        initializer (list[str]): List of initializers to use.
        beta (list[float]): List of beta values to use.
        samples (int): Number of samples from the parameter grid to use. If None, all combinations are used.

    Returns: Parameter grid as a dataframe.
    """
    if conditioning is None:
        conditioning = [True]
    if h_dim is None:
        h_dim = [[512, 256]]
    if z_dim is None:
        z_dim = [10]
    if dropout is None:
        dropout = [0.2]
    if activation is None:
        activation = ['relu']
    if initializer is None:
        initializer = ['glorot_normal']
    if beta is None:
        beta = [0.001]

    kwargs_list = list(items for _, items in kwargs.items())
    param_combinations = itertools.product(conditioning, h_dim, z_dim, dropout, activation,
                                           initializer, beta, *kwargs_list)
    arg_names = list(inspect.signature(create_param_df).parameters.keys())
    arg_names.remove('kwargs')
    arg_names.remove('samples')
    for key, _ in kwargs.items():
        arg_names.append(key)
    param_df = pd.DataFrame(param_combinations, columns=[arg_names])

    if samples is not None:
        param_df = param_df.sample(samples)

    return param_df


class VAECrossValidatorDF:
    """ Class for cross validating VAE models. Uses a dataframe to store the parameters for cross validation and 
    results.

    Args:
        param_df (pd.DataFrame): Dataframe containing the parameters to test.
            Note that the cross validation accepts some additional parameters to the DataFrame default. Currently
            supported additional parameters are: \n
                - normalization (int): Normalization to use for the data. Must be one of 0, 1, 2 or 3. Definitions are found in utils \n
                - subset (str): Subset of the data to use. Must be one of 'all', 'thickness', 'volume', or 'thickness_volume'. \n
                - batch_size (int): Batch size to use for training. \n
                - lr (float): Learning rate to use for training. \n
                - epochs (int): Number of epochs to train for. \n
                - lr_scheduler (tf.keras.callbacks.LearningRateScheduler): Learning rate scheduler to use for training. \n
        k_folds (int): Number of folds to use for cross validation.
        save_path (str): Path to save the models to.
        test_mode (bool): If True, the cross validation is run in test mode. This means that data is stored when function wrappers are
            called so the data can be inspected.
    """
    def __init__(self, param_df: pd.DataFrame, k_folds: int = 5, save_path: str = None, test_mode: bool = False):
        self.param_df = param_df
        self.save_path = save_path
        self.kf = KFold(n_splits=k_folds)
        self.generate_data_params = [] # For testing purposes
        self.train_val_vae_params = [] # For testing purposes
        self.test_mode = test_mode # For testing purposes
    
    def cross_validate(self, datapath: str, verbose: int = 1) -> pd.DataFrame:
        """ Run a cross validation for k folds over the data provided in the given dataframe.

        Args:
            datapath (str): Path to the data to use for cross validation.
            verbose (int): Verbosity level passed to fit.
              0 = progress bar on cross validations, 1 = progress bar on folds for current model, 2 = metrics for model training

        Returns: A dataframe containing the parameters tested and the results of the cross validation.
        """

        data_sets = {}
        self.generate_data_params.clear()
        if "normalization" in self.param_df.columns and "subset" in self.param_df.columns:
            norms = [norm[0] for norm in self.param_df["normalization"].to_numpy()]
            subsets = [subset[0] for subset in self.param_df["subset"].to_numpy()]
            for norm, subset in set(zip(norms, subsets)):
                train, val, test, labels = self.generate_data_wrapper(filepath=datapath, normalize=norm, subset=subset)
                data_sets[(norm, subset)] = (train, val, test, labels)
        elif "normalization" in self.param_df.columns:
            norms = [norm[0] for norm in self.param_df["normalization"].to_numpy()]
            for norm in set(norms):
                train, val, test, labels = self.generate_data_wrapper(filepath=datapath, normalize=norm)
                data_sets[norm] = (train, val, test, labels)
        elif "subset" in self.param_df.columns:
            subsets = [subset[0] for subset in self.param_df["subset"].to_numpy()]
            for subset in subsets:
                    train, val, test, labels = self.generate_data_wrapper(filepath=datapath, subset=subset)
                    data_sets[subset] = (train, val, test, labels)    
        else:
            train, val, test, labels = self.generate_data_wrapper(filepath=datapath)
            

        for j, row in tqdm(self.param_df.iterrows(), total=len(self.param_df), desc="Model Progress", ncols=80):
            best_cv_total_loss = []
            best_cv_recon_loss = []
            best_cv_kl_loss = []
            best_cv_r2 = []
            cv_total_hist = []
            cv_recon_hist = []
            cv_kl_hist = []
            training_total_hist = []
            training_recon_hist = []
            training_kl_hist = []
            validation_feat_r2 = []

            if "normalization" in self.param_df.columns and "subset" in self.param_df.columns:
                train, val, test, labels = data_sets[(row["normalization"], row["subset"])]
            elif "normalization" in self.param_df.columns:
                train, val, test, labels = data_sets[row["normalization"]]
            elif "subset" in self.param_df.columns:
                train, val, test, labels = data_sets[row["subset"]]
                
            if "batch_size" in self.param_df.columns:
                data = train.shuffle(10000).batch(row["batch_size"])
            else:
                data = train.shuffle(10000).batch(256)
            val_data = val.batch(val.cardinality().numpy())

            for i in tqdm(range(self.kf.n_splits), desc="Fold Progress", ncols=80, disable = True if verbose < 1 else False):
                vae = None
                input_dim = train.element_spec[0].shape[0]
                cov_dim = train.element_spec[1].shape[0]
                if(row["conditioning"]):
                    encoder = create_vae_encoder(input_dim + cov_dim, row["h_dim"], row["z_dim"], row["activation"], 
                                                row["initializer"], row["dropout"])
                    decoder = create_vae_decoder(row["z_dim"] + cov_dim, row["h_dim"], input_dim, row["activation"],
                                                row["initializer"], row["dropout"])
                    vae = VAE(encoder, decoder, row['beta'], cov=True)
                else:
                    encoder = create_vae_encoder(input_dim, row["h_dim"], row["z_dim"], row["activation"], 
                                                row["initializer"], row["dropout"])
                    decoder = create_vae_decoder(row["z_dim"], row["h_dim"], input_dim, row["activation"],
                                                row["initializer"], row["dropout"])
                    vae = VAE(encoder, decoder, row['beta'], cov=False)

                if("lr" in self.param_df.columns):
                    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=row["lr"]))
                else:
                    vae.compile()
                
                cv_val_data = data.shard(self.kf.n_splits, i)
                train_data = data.shard(self.kf.n_splits, (i+1) % self.kf.n_splits)

                for k in range(2, self.kf.n_splits):
                    train_data = train_data.concatenate(data.shard(self.kf.n_splits, (i+k) % self.kf.n_splits))

                if "lr_scheduler" in self.param_df.columns:
                    callbacks = [row["lr_scheduler"]]
                else:
                    callbacks = []

                if "epochs" in self.param_df.columns:
                    vae, hist = self.train_val_vae_wrapper(vae=vae, train_data=train_data, val_data=cv_val_data, verbose=verbose-1, epochs=row["epochs"], callbacks=callbacks)
                else:
                    vae, hist = self.train_val_vae_wrapper(vae=vae, train_data=train_data, val_data=cv_val_data, verbose=verbose-1, epochs=200, callbacks=callbacks)

                # Top metrics from each fold
                best_cv_total_loss.append(np.min(hist.history['val_total_loss']))
                best_cv_recon_loss.append(np.min(hist.history['val_reconstruction_loss']))
                best_cv_kl_loss.append(np.min(hist.history['val_kl_loss']))
                mean_r2 = np.mean(hist.history['val_r2_feat'], axis=1)
                best_cv_r2.append(hist.history['val_r2_feat'][np.argmax(mean_r2)])
                
                # History from each fold
                cv_total_hist.append(hist.history['val_total_loss'])
                cv_recon_hist.append(hist.history['val_reconstruction_loss'])
                cv_kl_hist.append(hist.history['val_kl_loss'])
                training_total_hist.append(hist.history['total_loss'])
                training_recon_hist.append((hist.history['reconstruction_loss']))
                training_kl_hist.append(hist.history['kl_loss'])

                # Fixed validation set R2 performance from each fold
                val_dict = vae.evaluate(val_data, verbose=verbose-1, return_dict=True)
                validation_feat_r2.append(val_dict['r2_feat'])
            
            # Average metrics from all folds
            row["avg_best_cv_total_loss"] = np.mean(best_cv_total_loss)
            row["avg_best_cv_recon_loss"] = np.mean(best_cv_recon_loss)
            row["avg_best_cv_kl_loss"] = np.mean(best_cv_kl_loss)
            row["avg_best_cv_r2"] = np.mean(best_cv_r2, axis=0)
            row["avg_cv_total_loss_history"] = np.mean(cv_total_hist, axis=0)
            row["avg_cv_recon_loss_history"] = np.mean(cv_recon_hist, axis=0)
            row["avg_cv_kl_loss_history"] = np.mean(cv_kl_hist, axis=0)
            row["avg_training_total_loss_history"] = np.mean(training_total_hist, axis=0)
            row["avg_training_recon_loss_history"] = np.mean(training_recon_hist, axis=0)
            row["avg_training_kl_loss_history"] = np.mean(training_kl_hist, axis=0)
            row["avg_val_feature_r2"] = np.mean(validation_feat_r2, axis=0)
            
            # Instantiate a new dataframe to store the results
            if j == 0:
                results = pd.DataFrame(columns=row.keys())

            results.loc[len(results)] = row
            # saves checkpoint for after each cross validation
            if self.save_path is not None:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                with open(os.path.join(self.save_path, 'results.pkl'), 'wb') as f:
                    pickle.dump(results, f)
        return results
    
    def generate_data_wrapper(self, **kwargs):
        """ Wrapper for generate_data that stores the parameters when generate_data is called
         This method is used for testing.
          
        Args:
            **kwargs: Keyword arguments to pass to generate_data.
             
        Returns: The output of generate_data.
        """
        if self.test_mode:
            self.generate_data_params.append(kwargs)
        return generate_data(**kwargs)
    
    def train_val_vae_wrapper(self, **kwargs):
        """ Wrapper for train_val_vae that stores the parameters when train_val_vae is called
         This method is used for testing.
          
        Args:
            **kwargs: Keyword arguments to pass to train_val_vae.
             
        Returns: The output of train_val_vae.
        """
        if self.test_mode:
            self.train_val_vae_params.append(kwargs)
        return train_val_vae(**kwargs)


class CyclicAnnealingBeta:
    """Scheduler that implements the cyclical annealing beta schedule as described in the paper.

     Args:
        step_size (int): Number of training iterations per half cycle.
        cycles (int): Number of cycles (default M = 4).
        proportion (float): Proportion used to increase beta within a cycle (default R = 0.5).

    Attributes:
        beta (float): Current beta value.
        step_size (int): Number of training iterations per half cycle.
        cycles (int): Number of cycles.
        proportion (float): Proportion used to increase beta within a cycle.
        iteration (int): Current iteration.
     """
    def __init__(self, step_size, cycles=4, proportion=0.5, max_beta=1e-3):
        self.beta = 0
        self.step_size = step_size
        self.cycles = cycles
        self.proportion = proportion
        self.iteration = 0
        self.max_beta = max_beta

    def get_beta(self):
        cycle_length = 2 * self.step_size
        cycle_number = (self.iteration // cycle_length) % self.cycles
        phase = (self.iteration % cycle_length) / cycle_length
        if phase < self.proportion:
            beta = self.max_beta * (phase / self.proportion)
        else:
            beta = self.max_beta

        return beta

    def step(self):
        self.beta = self.get_beta()
        self.iteration += 1
        return self.beta


class CyclicalAnnealingBetaCallback(Callback):
    """Custom callback to update the beta value in the VAE model based on the cyclical annealing beta scheduler.

    Args:
        scheduler (CyclicAnnealingBeta): The cyclical annealing beta scheduler.

    Attributes:
        scheduler (CyclicAnnealingBeta): The cyclical annealing beta scheduler.
    """
    def __init__(self, scheduler):
        super(CyclicalAnnealingBetaCallback, self).__init__()
        self.scheduler = scheduler

    def on_batch_begin(self, batch, logs=None):
        # Update the beta value in the VAE model based on the scheduler
        new_beta = self.scheduler.step()
        self.model.beta.assign(new_beta)
#%%
