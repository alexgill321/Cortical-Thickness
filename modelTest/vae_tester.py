from utils import generate_data
from modelUtils.vae_utils import train_vae, test_vae
import os


def main():
    """ Generate and train VAE models with desired parameters
    """
    # -------------------------------------------------------------------------
    h_dim = [100, 100]
    z_dim = 20
    batch_size = 64
    epochs = 2000
    lr = 0.0001
    l2_reg = 0.001
    # -------------------------------------------------------------------------
    # Generate data
    cur = os.getcwd()
    filepath = os.path.join(cur, '../outputs/megasample_cleaned.csv')

    # TODO: Ensure that data is being regularized correctly
    # TODO: Add noise as in paper to training data
    train_data, test_data = generate_data(filepath)

    # Train VAE
    save_dir = os.path.join(cur, '../outputs/models/vae/vae1/')
    train_vae(train_data, batch_size, epochs, lr, h_dim, z_dim, l2_reg, save_dir)
    results = test_vae(test_data, save_dir)
    print(results)


if __name__ == "__main__":
    main()
