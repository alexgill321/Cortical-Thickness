import tensorflow as tf
from models.vae_models import VAE, VAEEncoder, VAEDecoder
from sklearn.model_selection import KFold
import os


def save_vae(vae, savefile):
    """ Save a VAE model.

    Args:
        vae (VAE): VAE model to be saved.
        savefile (str): Path to save the model to.
    """
    if not os.path.exists(savefile):
        os.makedirs(savefile)
    vae.encoder.save(os.path.join(savefile, 'encoder'), save_format="tf")
    vae.decoder.save(os.path.join(savefile, 'decoder'), save_format="tf")


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
    encoder = VAEEncoder(h_dim, z_dim)
    decoder = VAEDecoder(h_dim, n_features)
    vae = VAE(encoder, decoder)
    return vae


def train_val_vae(vae, data, batch_size=256, epochs=200, save=False):
    """ Train and validate a VAE model.

    Trains and validates a VAE model on the given training and validation data. The model is trained for the given
    number of epochs and the best model is saved if save=True.

    Args:
        vae (VAE): VAE model to train. Needs to be compiled with the desired optimizers, loss functions and callbacks.
        data
        batch_size (int): Batch size.
        epochs (int): Number of epochs to train for.
        save (bool): If True, the best model is saved.

    Returns: Best model, as determined by the lowest validation loss.
    """

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    vae.fit(data.batch(batch_size), epochs=epochs, validation_split=0.2, callbacks=[stop_early])

    if save:
        save_vae(vae, save)


    return vae


def cross_validate_vae(data, k=5, param_dict=None, batch_size=256, epochs=200):
    """Perform k-fold cross-validation for VAE model with different parameter combinations.

    Args:
        data (tf.data.Dataset): Dataset to perform cross-validation on.
        k (int, optional): Number of folds for cross-validation.


    Returns:
        A tuple containing the best parameters and their corresponding average validation loss.
    """
    # Convert the dataset to a list of (x, y) pairs
    if lr_values is None:
        lr_values = [0.0001]
    if h_dim_values is None:
        h_dim_values = [[100, 100]]
    if z_dim_values is None:
        z_dim_values = [20]
    data_list = list(data.as_numpy_iterator())

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=k)
    n_features = data_list[0][0].shape[1]
    best_params = None
    best_loss = float('inf')

    # Iterate over all combinations of parameters
    for lr in lr_values:
        for h_dim in h_dim_values:
            for z_dim in z_dim_values:
                total_loss = 0

                # Perform k-fold cross-validation
                for train_indices, val_indices in kf.split(data_list):
                    train_data = tf.data.Dataset.from_tensor_slices([data_list[i] for i in train_indices])
                    val_data = tf.data.Dataset.from_tensor_slices([data_list[i] for i in val_indices])

                    # Create and train the VAE model with the current parameter combination
                    vae = create_vae(n_features, h_dim, z_dim)
                    vae_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                    vae.compile(optimizer=vae_optimizer)
                    vae.fit(train_data.batch(batch_size), epochs=epochs, validation_data=val_data.batch(batch_size),
                            verbose=0)

                    # Compute the validation loss
                    val_loss = vae.evaluate(val_data.batch(batch_size), verbose=0)

                    total_loss += val_loss

                # Calculate the average validation loss for the current parameter combination
                avg_loss = total_loss / k

                # Update the best parameters and loss if necessary
                if avg_loss < best_loss:
                    best_params = (lr, h_dim, z_dim)
                    best_loss = avg_loss

    return best_params, best_loss


def create_vae(n_features, h_dim, z_dim):
    encoder = VAEEncoder(h_dim, z_dim)
    decoder = VAEDecoder(h_dim, n_features)
    vae = VAE(encoder, decoder)
    return vae

#%%
