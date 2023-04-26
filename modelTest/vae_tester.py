from utils import data_validation
from modelUtils.vae_utils import train_val_vae, test_vae_from_file, train_vae
import os
import numpy as np


def main():
    """ Generate and train VAE models with desired parameters
    """
    # -------------------------------------------------------------------------
    h_dim = [100]
    z_dim = 20
    batch_size = 64
    epochs = 2000
    lr = 0.00014
    # -------------------------------------------------------------------------
    # Generate data
    cur = os.getcwd()
    filepath = os.path.join(cur, '../outputs/megasample_cleaned.csv')

    train_data, val_data, test_data = data_validation(filepath)
    test_batch_size = test_data.element_spec[0].shape[0]
    # Train VAE
    save_dir = os.path.join(cur, '../outputs/models/vae/vae1/')
    train_val_vae(train_data, val_data, batch_size, epochs, lr, h_dim, z_dim, save_dir)
    results = test_vae_from_file(test_data, save_dir, test_batch_size)


if __name__ == "__main__":
    main()
