#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from utils import data_validation
from models.vae_models import VAE, create_vae_decoder, create_vae_encoder
import seaborn as sns
import tensorflow as tf

# Assuming you have already trained your VAE model
# and have a dataset (X) to visualize


#%% md
# Training VAE
#%%
h_dim = [300, 300]
z_dim = 20
batch_size = 128
epochs = 300
beta = 0.0001
lr = 0.005
dropout = 0.2
# -------------------------------------------------------------------------
# Generate data
cur = os.getcwd()
filepath = os.path.join(cur, 'outputs/megasample_cleaned.csv')

train_data, val_data, test_data = data_validation(filepath, validation_split=0.2)
test_batch_size = test_data.element_spec[0].shape[0]
save_dir = os.path.join(cur, 'outputs/models/vae/data_vis/')
encoder = create_vae_encoder(input_dim=train_data.element_spec[0].shape[0], hidden_dim=h_dim, latent_dim=z_dim,
                             dropout_rate=dropout)
decoder = create_vae_decoder(latent_dim=z_dim, hidden_dim=h_dim, output_dim=train_data.element_spec[0].shape[0],
                             dropout_rate=dropout)
vae = VAE(encoder, decoder, beta=beta)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
train_data = train_data.batch(batch_size)
vae.fit(train_data, epochs=epochs)

#%%
val_batch_size = val_data.cardinality().numpy()
val_data = val_data.batch(val_batch_size)
val_batch = next(iter(val_data))
# 1. Encode input samples
z_mean, z_log_var, z, _ = vae(val_batch)
z = z.numpy()

# 2. Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
z_2d = tsne.fit_transform(z)

# 3. Create a scatter plot
plt.scatter(z_2d[:, 0], z_2d[:, 1], cmap='viridis', alpha=0.6)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE Visualization of Latent Space")
plt.show()
#%%

# 2. Create subplots for each latent dimension
n_latent_dims = z.shape[1]
n_cols = 3
n_rows = n_latent_dims // n_cols + int(n_latent_dims % n_cols > 0)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten()

for i in range(n_latent_dims):
    sns.histplot(z[:, i], ax=axes[i], kde=True, color='blue', alpha=0.6, bins=20)
    axes[i].set_title(f'Latent Dimension {i + 1}')

# Remove any extra subplots if present
for i in range(n_latent_dims, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()
#%%

#%%
