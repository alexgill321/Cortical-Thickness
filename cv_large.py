"""
Run the main of this script to generate and run the large cross validation.

This cross validation encompasses a wide range of hyperparameters, including:

- The number of layers in the encoder and decoder
- The number of neurons in each layer
- The activation function for each layer
- The initializer for each layer
- The dropout rate for each layer
- The latent dimensionality
- The batch size
- The number of epochs
- The learning rate
- covariances in the encoder and decoder
- data normalization method
- data subset (thickness, volume, thickness_volume, all)
"""

import argparse
import os
from modelUtils.vae_utils import create_param_df, VAECrossValidatorDF
import tensorflow as tf


def __main__():
    parser = argparse.ArgumentParser(description='Run the large cross validation')
    parser.add_argument('--output_dir', type=str, help='Directory where model results will be saved')
    parser.add_argument('--file_path', type=str, help='Path to the data file')
    args = parser.parse_args()

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    file_path = args.file_path
    output_dir = args.output_dir

    # Create the parameter dataframe
    param_df = create_param_df(
        conditioning=[False],
        z_dim=[2, 3, 5, 10, 20],
        batch_size=[128],
        epochs=[300],
        learning_rate=[1e-4],
        dropout=[0.1],
        normalization=[1], subset=['thickness'],
        beta=[1e-5], h_dim=[[512, 256]]
        )

    # Create the cross validator
    cv = VAECrossValidatorDF(param_df, save_path=output_dir, k_folds=2)
    results = cv.cross_validate(datapath=file_path)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    results.to_csv(os.path.join(output_dir, 'cv_large_results_z.csv'))


if __name__ == '__main__':
    __main__()
#%%


