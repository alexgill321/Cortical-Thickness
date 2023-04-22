from utils import data_validation
from modelUtils.vae_utils import train_vae, test_vae
import os
import numpy as np


def main():
    """ Generate and train VAE models with desired parameters
    """
    # -------------------------------------------------------------------------
    h_dim = [100]
    z_dim = 20
    batch_size = 64
    epochs = 400
    lr = 0.001
    # -------------------------------------------------------------------------
    # Generate data
    cur = os.getcwd()
    filepath = os.path.join(cur, '../outputs/megasample_cleaned.csv')

    train_data, val_data, test_data = data_validation(filepath)

    # Train VAE
    save_dir = os.path.join(cur, '../outputs/models/vae/vae1/')
    train_vae(train_data, batch_size, epochs, lr, h_dim, z_dim, save_dir)
    results = test_vae(test_data, save_dir)
    print(np.mean(results))


if __name__ == "__main__":
    main()
