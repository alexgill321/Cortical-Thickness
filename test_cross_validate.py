from modelUtils.vae_utils import create_param_grid, VAECrossValidator
import os
from utils import data_validation
import matplotlib.pyplot as plt
#%%
cur = os.getcwd()
filepath = os.path.join(cur, 'outputs/megasample_cleaned.csv')
train_data, val_data, test_data = data_validation(filepath)
input_dim = train_data.element_spec[0].shape[0]

#%%
# cross validation for the beta parameter
beta = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
epochs = 200
beta_grid = create_param_grid([[100, 100]], [30], [0.2], ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(beta_grid, input_dim, 5, batch_size=128)
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

best_val_loss = 1000
best_params = None
for res in results:
    metrics = res[1]
    params = res[0]
    label = ''
    if metrics['recon_loss'] < best_val_loss:
        best_val_loss = metrics['recon_loss']
        best_params = params['vae']['beta']
    label = str(params['vae']['beta'])
    axs[0].plot(range(epochs), metrics['avg_training_losses'], label=label)
    axs[1].plot(range(epochs), metrics['avg_kl_losses'], label=label)

axs[0].legend()
axs[0].set_title('Beta vs Training Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Training Loss')

axs[1].legend()
axs[1].set_title('Beta vs KL Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('KL Loss')

plt.tight_layout()  # Adjusts subplot parameters to give specified padding
plt.savefig('outputs/Images/beta_vs_loss.png')

#%%
print(f'Best params: {best_params}, Best val loss: {best_val_loss}')

#%%
h_dims = [[50, 50], [100, 100], [150, 150], [200, 200], [250, 250], [300, 300], [150, 100, 50], [200, 150, 100],
          [250, 200, 150], [300, 200, 100]]

h_dim_grid = create_param_grid(h_dims, [30], [0.2], ['relu'], ['glorot_uniform'], [.001])
#%%
cv = VAECrossValidator(h_dim_grid, input_dim, 5, batch_size=128)
#%%
results = cv.cross_validate(train_data, epochs=50, verbose=0)
#%%
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

best_val_loss = 1000
best_params = None
for res in results:
    metrics = res[1]
    params = res[0]
    label = ''
    if metrics['recon_loss'] < best_val_loss:
        best_val_loss = metrics['recon_loss']
        best_params = params['encoder']['hidden_dim']
    for dim in params['encoder']['hidden_dim']:
        label += str(dim) + '_'
    axs[0].plot(range(50), metrics['avg_training_losses'], label=label)
    axs[1].plot(range(50), metrics['avg_kl_losses'], label=label)

axs[0].legend()
axs[0].set_title('Hidden Layer Dimensions vs Training Reconstruction Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Training Loss')

axs[1].legend()
axs[1].set_title('Hidden Layer Dimensions vs KL Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('KL Loss')

plt.tight_layout()  # Adjusts subplot parameters to give specified padding
plt.savefig('outputs/Images/h_dim_vs_loss.png')
#%%
print(f'Best params: {best_params}, Best val loss: {best_val_loss}')

#%%
latent_dims = [5, 10, 15, 20, 25, 30]
latent_dim_grid = create_param_grid([[300, 300]], latent_dims, [0.2], ['relu'], ['glorot_uniform'], [.001])

#%%
cv = VAECrossValidator(latent_dim_grid, input_dim, 5, batch_size=128)
#%%
results = cv.cross_validate(train_data, epochs=50, verbose=0)

#%%
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

best_val_loss = 1000
best_params = None
for res in results:
    metrics = res[1]
    params = res[0]
    label = ''
    if metrics['recon_loss'] < best_val_loss:
        best_val_loss = metrics['recon_loss']
        best_params = params['encoder']['latent_dim']
    label = str(params['encoder']['latent_dim'])
    axs[0].plot(range(50), metrics['avg_training_losses'], label=label)
    axs[1].plot(range(50), metrics['avg_kl_losses'], label=label)

axs[0].legend()
axs[0].set_title('Latent Dimension vs Training Reconstruction Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Training Loss')

axs[1].legend()
axs[1].set_title('Latent Dimension vs KL Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('KL Loss')

plt.tight_layout()  # Adjusts subplot parameters to give specified padding
plt.savefig('outputs/Images/latent_dim_vs_loss.png')

#%%
print(f'Best params: {best_params}, Best val loss: {best_val_loss}')
