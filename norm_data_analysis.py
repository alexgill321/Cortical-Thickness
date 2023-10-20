from modelUtils.vae_utils import train_val_vae, CyclicAnnealingBeta, CyclicalAnnealingBetaCallback
from vaeModelAnalyzer import VAEModelAnalyzer
from models.vae_models import create_vae_decoder, create_vae_encoder, VAE
from utils import generate_data_thickness_only
from modelUtils.lr_utils import MultiOptimizerLearningRateScheduler, CyclicLR, ExponentialDecayScheduler
import os
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.image import imread
import vis_utils as vu
import numpy as np

cur = os.getcwd()
#%%
filepath = os.path.join(cur, 'data/cleaned_data/megasample_cleaned.csv')
train_data, val_data, test_data, feat_labels = generate_data_thickness_only(filepath, normalize=1)
num_features = train_data.element_spec[0].shape[0]
cov_dim = train_data.element_spec[1].shape[0]
train_data = train_data.batch(64)
val_data = val_data.batch(val_data.cardinality().numpy())
test_data = test_data.batch(test_data.cardinality().numpy())
#%%
"""
betas = [1e-4, 1e-5, 1e-6]

for beta in betas:
    encoder = create_vae_encoder(input_dim=num_features, hidden_dim=[100, 100], latent_dim=10)
    decoder = create_vae_decoder(latent_dim=10, hidden_dim=[100, 100], output_dim=num_features)
    vae = VAE(encoder, decoder, beta=beta)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    model, hist = train_val_vae(vae, train_data, val_data, epochs=300, verbose=1)
    analyzer = VAEModelAnalyzer(model, next(iter(val_data)), 10, feat_labels, hist=hist)
    save_path = os.path.join(cur, f'outputs/analysis/no_norm_data_beta_{beta}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    analyzer.full_stack(save_path)
    plt.figure()
    plt.axis('off')
    plt.imshow(imread(save_path + '/full_stack.png'))
    plt.show()
    plt.close()
"""
#%%
beta = 1e-5
h_dim = [512, 256]
z_dim = [5]
base_lr = 1e-3
max_lr = 1e-6
step_size = 50
decay_steps = 10
decay_rate = 0.88

# Create the cyclical annealing beta scheduler
# beta_scheduler = CyclicAnnealingBeta(step_size=500, proportion=0.6, max_beta=1e-4)

# opt_scheduler = [CyclicLR(base_lr, step_size, max_lr, mode='triangular')]
# opt_scheduler = [ExponentialDecayScheduler(max_lr, decay_rate, decay_steps)]

for z in z_dim:
    encoder = create_vae_encoder(input_dim=num_features+cov_dim, hidden_dim=h_dim, latent_dim=z, dropout_rate=0.2)
    decoder = create_vae_decoder(latent_dim=z+cov_dim, hidden_dim=h_dim, output_dim=num_features, dropout_rate=0.2)
    vae = VAE(encoder, decoder, beta=beta, cov=True)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    optimizer = [vae.optimizer]

    # lr_and_opt = zip(opt_scheduler, optimizer)
    # scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)
    # scheduler = CyclicalAnnealingBetaCallback(beta_scheduler)
    model, hist = train_val_vae(vae, train_data, val_data, epochs=400, verbose=1)
    analyzer = VAEModelAnalyzer(model, next(iter(val_data)), z, feat_labels, hist=hist, test_data=next(iter(test_data)))
    save_path = os.path.join(cur, f'outputs/analysis/cov_sn_z_{z}')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    analyzer.full_stack(save_path)

    # without covariates
    encoder = create_vae_encoder(input_dim=num_features, hidden_dim=h_dim, latent_dim=z, dropout_rate=0.2)
    decoder = create_vae_decoder(latent_dim=z, hidden_dim=h_dim, output_dim=num_features, dropout_rate=0.2)
    vae_no_cov = VAE(encoder, decoder, beta=beta, cov=False)
    vae_no_cov.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # lr_and_opt = zip(opt_scheduler, optimizer)
    # scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)
    # scheduler = CyclicalAnnealingBetaCallback(beta_scheduler)
    model, hist = train_val_vae(vae_no_cov, train_data, val_data, epochs=400, verbose=1)
    analyzer = VAEModelAnalyzer(model, next(iter(val_data)), z, feat_labels, hist=hist, test_data=next(iter(test_data)))
    save_path = os.path.join(cur, f'outputs/analysis/no_cov_sn_z_{z}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    analyzer.full_stack(save_path)
