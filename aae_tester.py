from utils import generate_data
from generate_models import train_aae
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
    cur = os.getcwd()
    filepath = os.path.join(cur, 'outputs/megasample_cleaned.csv')
    data = generate_data(filepath)
    print(data.element_spec[0].shape[1])
    aae1 = train_aae(data, batch_size, epochs, lr, h_dim, z_dim)
    save_dir = os.path.join(cur, 'outputs/models/aae/aae1')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    aae1.encoder.save_weights(os.path.join(save_dir, 'encoder.h5'))
    aae1.decoder.save_weights(os.path.join(save_dir, 'decoder.h5'))
    aae1.discriminator.save_weights(os.path.join(save_dir, 'discriminator.h5'))


if __name__ == "__main__":
    main()

#%%

#%%
