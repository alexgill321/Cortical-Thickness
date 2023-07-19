""" The purpose of this script is to perform a loose optimization of the cortical thickness data run through a vae
before performing a more rigorous optimization. This is done to get a rough idea of the best hyperparameters to use for
a cross validation. In this case we use a latent dimensionality of 15.
"""

from utils import generate_data_thickness_only
from models.vae_models import create_vae_decoder, create_vae_encoder, VAE
from modelUtils.vae_utils import create_vae
import matplotlib.pyplot as plt
import os

cur = os.getcwd()
#%% Generate data
filename = 'megasample_ctvol_500sym_max2percIV_cleaned.csv'
filepath = os.path.join(cur, '../data/cleaned_data/', filename)
train_data, val_data, test_data = generate_data_thickness_only(filepath)
num_features = train_data.element_spec[0].shape[0]
train_data = train_data.batch(128)

val_batch_size = val_data.cardinality().numpy()
val_data = val_data.batch(val_batch_size)

#%% Test performance for different beta values
betas = [1e-2, 1e-3, 1e-4, 1e-5]
hist = []
val_results = []

for beta in betas:
    encoder = create_vae_encoder(num_features, [100, 50], 15, 'relu', 'glorot_uniform')
    decoder = create_vae_decoder(15, [50, 100], num_features, 'relu', 'glorot_uniform')
    vae = VAE(encoder, decoder, beta)
    vae.compile()

    history = vae.fit(train_data, epochs=300, validation_data=val_data, verbose=0)
    hist.append(history)
    vr = vae.evaluate(val_data, verbose=0)
    val_results.append(vr)

#%% Visualize Results
for i, history in enumerate(hist):
    beta = betas[i]

    # Plot Reconstruction Loss
    plt.figure()
    plt.plot(history.history['reconstruction_loss'], label='Training Loss')
    plt.plot(history.history['val_reconstruction_loss'], label='Validation Loss')
    plt.title(f'Reconstruction Loss (Beta={beta})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot KL Loss
    plt.figure()
    plt.plot(history.history['kl_loss'], label='KL Loss')
    plt.plot(history.history['val_kl_loss'], label='Validation KL Loss')
    plt.title(f'KL Loss (Beta={beta})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

