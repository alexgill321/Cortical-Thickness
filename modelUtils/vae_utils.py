import tensorflow as tf
from models.vae_models import VAE, VAEEncoder, VAEDecoder
from sklearn.model_selection import KFold
import os
import numpy as np
from keras.callbacks import LambdaCallback


def train_vae(data, batch_size=256, epochs=200, lr=0.0001, h_dim=None, z_dim=20, savefile=None):
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
        savefile (str, optional):
         Path to save the model to.
    """
    if h_dim is None:
        h_dim = [100, 100]

    # Batch data
    data = data.batch(batch_size)
    n_features = data.element_spec[0].shape[1]
    n_labels = data.element_spec[1].shape[1]

    # Create vae model
    vae = create_vae(n_features, h_dim, z_dim)

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
    return vae

def train_val_vae(train_data, val_data, batch_size=256, epochs=200, lr=0.0001, h_dim=None, z_dim=20, savefile=None):
    """Create, train, and save a VAE model

    Args:
        train_data (tf.data.Dataset):
         Dataset to train the model on.
        val_data (tf.data.Dataset):
         Dataset to validate the model on.
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
        savefile (str, optional):
         Path to save the model to.
    """
    if h_dim is None:
        h_dim = [100, 100]

    # Batch data
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)
    n_features = train_data.element_spec[0].shape[1]
    n_labels = train_data.element_spec[1].shape[1]

    # Create vae model
    vae = create_vae(n_features, h_dim, z_dim)

    # Create optimizer
    vae_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Compile model
    vae.compile(vae_optimizer)

    # Train model
    vae.fit(train_data, epochs=epochs, validation_data=val_data)

    # Save model
    if savefile is not None:
        if not os.path.exists(savefile):
            os.makedirs(savefile)
        vae.encoder.save(os.path.join(savefile, 'encoder'), save_format="tf")
        vae.decoder.save(os.path.join(savefile, 'decoder'), save_format="tf")


def test_vae_from_file(test_data, savefile, batch_size=256):
    """Evaluate a VAE model over a dataset.

    Args:
        test_data (tf.data.Dataset):
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
    test_data = test_data.batch(batch_size)
    output = vae.evaluate(test_data)
    return output


def cross_validate_vae(data, k=5, batch_size=256, epochs=200, lr_values=[0.0001], h_dim_values=[[100, 100]], z_dim_values=[20]):
    """Perform k-fold cross-validation for VAE model with different parameter combinations.

    Args:
        data (tf.data.Dataset): Dataset to perform cross-validation on.
        k (int, optional): Number of folds for cross-validation.
        batch_size (int, optional): Batch size to use for training.
        epochs (int, optional): Number of epochs to train for.
        lr_values (list, optional): List of learning rate values to test.
        h_dim_values (list, optional): List of hidden layer dimension combinations to test.
        z_dim_values (list, optional): List of latent space dimension values to test.

    Returns:
        A tuple containing the best parameters and their corresponding average validation loss.
    """
    # Convert the dataset to a list of (x, y) pairs
    data_list = list(data.as_numpy_iterator())

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=k)

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
                    vae.compile(vae_optimizer)
                    vae.fit(train_data.batch(batch_size), epochs=epochs, validation_data=val_data.batch(batch_size), verbose=0)

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


def find_learning_rate(train_data, batch_size=256, h_dim=None, z_dim=20, min_lr=1e-8, max_lr=1.0, n_steps=100):
    """Find the optimal learning rate for the VAE model.

    Args:
        train_data (tf.data.Dataset):
         Dataset to train the model on.
        batch_size (int, optional):
         Batch size to use for training.
        h_dim (list, optional):
         List of hidden layer dimensions.
        z_dim (int, optional):
         Dimension of latent space.
        min_lr (float, optional):
         Minimum learning rate to use for the search.
        max_lr (float, optional):
         Maximum learning rate to use for the search.
        n_steps (int, optional):
         Number of steps to run the learning rate finder.

    Returns:
        A tuple containing the learning rates and corresponding losses.
    """
    if h_dim is None:
        h_dim = [100, 100]

    # Batch data
    train_data = train_data.batch(batch_size).repeat()
    n_features = train_data.element_spec[0].shape[1]

    # Create VAE model
    vae = create_vae(n_features, h_dim, z_dim)

    # Create optimizer
    initial_lr = min_lr
    vae_optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

    # Compile model
    vae.compile(vae_optimizer)

    # Define learning rate update function
    def update_learning_rate(batch, logs):
        nonlocal initial_lr
        factor = np.exp(np.log(max_lr / min_lr) / n_steps)
        initial_lr = initial_lr * factor
        tf.keras.backend.set_value(vae_optimizer.lr, initial_lr)

    # Define learning rate update callback
    lr_update_callback = LambdaCallback(on_batch_end=update_learning_rate)

    # Store losses and learning rates
    rec_losses = []
    kl_losses = []
    lrs = []

    def save_losses(batch, logs):
        rec_losses.append(logs['reconstruction_loss'])
        kl_losses.append(logs['kl_loss'])
        lrs.append(tf.keras.backend.get_value(vae_optimizer.lr))

    # Define loss saving callback
    loss_saving_callback = LambdaCallback(on_batch_end=save_losses)

    # Train model with learning rate finder
    vae.fit(train_data, callbacks=[lr_update_callback, loss_saving_callback],
            steps_per_epoch=n_steps, epochs=1, verbose=0)

    return lrs, rec_losses, kl_losses


def create_vae(n_features, h_dim, z_dim):
    encoder = VAEEncoder(h_dim, z_dim)
    decoder = VAEDecoder(h_dim, n_features)
    vae = VAE(encoder, decoder)
    return vae

#%%
