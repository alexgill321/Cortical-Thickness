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

from modelUtils.vae_utils import create_param_df, VAECrossValidatorDF
import argparse
import os

def __main__():
    parser = argparse.ArgumentParser(description='Run the large cross validation')
    parser.add_argument('--output_dir', type=str, help='Directory where model results will be saved')
    parser.add_argument('--file_path', type=str, help='Path to the data file')
    args = parser.parse_args()

    file_path = args.file_path
    output_dir = args.output_dir

    # Create the parameter dataframe
    param_df = create_param_df(
        conditioning = [True, False], 
        z_dim = [2, 3, 4, 5, 10, 15, 20, 25, 30], 
        batch_size = [32, 64, 128, 256], 
        epochs = [500, 300, 200], 
        learning_rate = [1e-4, 1e-3, 1e-5], 
        dropout = [0.0, 0.1, 0.2, 0.3], 
        activation = ['relu', 'selu'], 
        initializer = ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'], 
        normalization = [0, 1, 2], subset = ['thickness', 'volume', 'thickness_volume', 'all'], 
        beta = [1e-4, 1e-3, 1e-5, 1e-6], h_dim = [[256, 128], [512, 256], [1024, 512], [128, 64], [256, 128, 64], [512, 256, 128], [1024, 512, 256]], 
        samples = 3000
        )

    # Create the cross validator
    cv = VAECrossValidatorDF(param_df, save_path = output_dir)
    results = cv.cross_validate(datapath=file_path)

    if not os.path.exist(output_dir):
        os.mkdir(output_dir)
    results.to_csv(os.path.join(output_dir, 'cv_large_results.csv'))


