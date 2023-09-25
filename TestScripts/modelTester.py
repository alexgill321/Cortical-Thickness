from utils import generate_data_thickness_only_analysis
from vaeModelAnalyzer import VAEModelAnalyzer
from models.vae_models import create_vae_decoder, create_vae_encoder, VAE
from modelUtils.lr_utils import MultiOptimizerLearningRateScheduler, CyclicLR, ExponentialDecayScheduler
from modelUtils.vae_utils import train_val_vae, CyclicAnnealingBeta, CyclicalAnnealingBetaCallback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model results will be saved')
parser.add_argument('--param_dir', type=str, help='Path to the parameter file')
args = parser.parse_args()

filepath = '../data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv'
train_data, val_data, test_data, feat_labels = generate_data_thickness_only_analysis(filepath, normalize=1)
