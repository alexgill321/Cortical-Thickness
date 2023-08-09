from modelUtils.vae_utils import train_val_vae, VAECrossValidator, create_param_grid, create_vae
from models.vae_models import create_vae_decoder, create_vae_encoder, VAE
from utils import generate_data_thickness_only
import os
import tensorflow as tf
import pickle as pkl

cur = os.getcwd()
#%%
filepath = os.path.join(cur, 'data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
train_data, val_data, test_data, feat_labels = generate_data_thickness_only(filepath, normalize=True)
num_features = train_data.element_spec[0].shape[0]
train_data = train_data.batch(128)
val_data = val_data.batch(val_data.cardinality().numpy())
#%%
betas = [1e-3, 1e-4, 1e-5]
histories = []
models = []
for beta in betas:
    encoder = create_vae_encoder(input_dim=num_features, hidden_dim=[100, 100], latent_dim=10)
    decoder = create_vae_decoder(latent_dim=10, hidden_dim=[100, 100], output_dim=num_features)
    vae = VAE(encoder, decoder, beta=beta)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    model, hist = train_val_vae(vae, train_data, val_data, epochs=300, verbose=1)
    models.append(model)
    histories.append(hist)
