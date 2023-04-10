from utils import generate_data
from modelUtils.aae_utils import train_aae, test_aae
import os


def main():
    """ Generate and train AAE models with desired parameters
    """
    # -------------------------------------------------------------------------
    h_dim = [100, 100]
    z_dim = 20
    batch_size = 256
    epochs = 200
    lr = 0.0001
    # -------------------------------------------------------------------------
    # Generate data
    cur = os.getcwd()
    filepath = os.path.join(cur, '../outputs/megasample_cleaned.csv')
    train_data, test_data = generate_data(filepath)

    # Train AAE
    save_dir = os.path.join(cur, '../outputs/models/aae/aae1/')
    train_aae(train_data, batch_size, epochs, lr, h_dim, z_dim, save_dir)
    results = test_aae(test_data, save_dir)
    print(results)


if __name__ == "__main__":
    main()

#%%

#%%
