#%% Imports
from modelUtils.vae_utils import create_param_grid, VAECrossValidator
import os
import pickle
from utils import generate_data_thickness_only
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modelUtils.vae_utils import load_or_train_model

cur = os.getcwd()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
save_path = cur + '/outputs/models/vae_df/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#%% Load Data
filepath = os.path.join(cur, 'data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
train_data, val_data, test_data, _ = generate_data_thickness_only(filepath)
val_batch_size = val_data.cardinality().numpy()
val_data_batched = val_data.batch(val_batch_size)
input_dim = train_data.element_spec[0].shape[0]

#%% Defining the Cross Validation
#latent_dims = [3, 4, 5, 10, 15, 20]
#beta = [0.001, 0.005, 0.01]
#hidden_dims = [[150, 100], [200, 150], [150, 150], [200, 200], [200, 100], [300, 150], [300, 200]]
dropout = [0.2]
epochs = 300
latent_dims = [20]
beta = [0.0001]
hidden_dims = [[100, 100]]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%% Running the Cross Validation
results = cv.cross_validate_df(train_data, epochs=epochs, verbose=1)
with open('outputs/CrossVal/cv_thickness_test.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%
with open('outputs/CrossVal/cv_thickness_norm.pkl', 'rb') as file:
    results = pickle.load(file)

#%% Defining the Heatmap Function


def create_heatmap_recon(data, **kwargs):
    heatmap_df = data.pivot(index='Beta', columns='Latent Dimensions', values='Reconstruction Loss')
    sns.heatmap(heatmap_df, annot=False, cmap='coolwarm', vmin=min_loss, vmax=max_loss, **kwargs)


def create_heatmap_kl(data, **kwargs):
    heatmap_df = data.pivot(index='Beta', columns='Latent Dimensions', values='KL Loss')
    sns.heatmap(heatmap_df, annot=False, cmap='coolwarm', vmin=min_loss, vmax=max_loss, **kwargs)


def create_heatmap_r2(data, **kwargs):
    heatmap_df = data.pivot(index='Beta', columns='Latent Dimensions', values='R2')
    sns.heatmap(heatmap_df, annot=False, cmap='coolwarm', vmin=min_loss, vmax=max_loss, **kwargs)


def create_heatmap_total(data, **kwargs):
    heatmap_df = data.pivot(index='Beta', columns='Latent Dimensions', values='Total Loss')
    sns.heatmap(heatmap_df, annot=False, cmap='coolwarm', vmin=min_loss, vmax=max_loss, **kwargs)


save_path = 'outputs/Images/CVAnalysis/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Convert the DataFrame Hidden Dimensions column to a string
results['Hidden Dimensions'] = results['Hidden Dimensions'].astype(str)

#%% Visualizing the Results for the Reconstruction Loss

# Determine the overall minimum and maximum reconstruction loss values
min_loss = results['Reconstruction Loss'].min()
max_loss = results['Reconstruction Loss'].max()

# Reshape the DataFrame using pivot
# Create a FacetGrid with hidden dimensions as the column variable
g = sns.FacetGrid(results, col='Hidden Dimensions')

# Map the function to each facet (hidden dimension)
g.map_dataframe(create_heatmap_recon)

# Set the labels for x-axis and y-axis
g.set_axis_labels('Latent Dimensions', 'Beta')

# Save the plot
plt.savefig(save_path + 'heatmap_recon_loss_v6.png')

#%% Visualizing the Results for the KL Loss

# Determine the overall minimum and maximum KL loss values
min_loss = results['KL Loss'].min()
max_loss = results['KL Loss'].max()

print(min_loss, max_loss)

# Create a FacetGrid with hidden dimensions as the column variable
g = sns.FacetGrid(results, col='Hidden Dimensions')

# Map the function to each facet (hidden dimension)
g.map_dataframe(create_heatmap_kl)

# Set the labels for x-axis and y-axis
g.set_axis_labels('Latent Dimensions', 'Beta')

# Save the plot
plt.savefig(save_path + 'heatmap_kl_loss_v6.png')

#%% Visualizing the Results for the R2

# Determine the overall minimum and maximum KL loss values
min_loss = results['R2'].min()
max_loss = results['R2'].max()

print(min_loss, max_loss)

# Create a FacetGrid with hidden dimensions as the column variable
g = sns.FacetGrid(results, col='Hidden Dimensions')

# Map the function to each facet (hidden dimension)
g.map_dataframe(create_heatmap_r2)

# Set the labels for x-axis and y-axis
g.set_axis_labels('Latent Dimensions', 'Beta')

# Save the plot
plt.savefig(save_path + 'heatmap_r2_v6.png')

#%% Visualizing the Results for the Total Loss

min_loss = results['Total Loss'].min()
max_loss = results['Total Loss'].max()

print(min_loss, max_loss)

# Create a FacetGrid with hidden dimensions as the column variable
g = sns.FacetGrid(results, col='Hidden Dimensions')

# Map the function to each facet (hidden dimension)
g.map_dataframe(create_heatmap_total)

# Set the labels for x-axis and y-axis
g.set_axis_labels('Latent Dimensions', 'Beta')

# Save the plot
plt.savefig(save_path + 'heatmap_total_loss_v6.png')

#%% Visualizing the Results for the Reconstruction Loss
save_path = 'outputs/Images/CVAnalysis/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


def plot_val_recon_loss(data, **kwargs):
    val_recon_loss = data['Validation Reconstruction Loss History'].values[0]
    e = len(val_recon_loss)
    epoch_list = list(range(1, e + 1))
    plt.plot(epoch_list, val_recon_loss, **kwargs)


# Create a FacetGrid with latent dimensions as the column variable
g = sns.FacetGrid(results, row='Hidden Dimensions', col='Latent Dimensions', hue='Beta', margin_titles=True)

# Map the function to each facet (latent dimension)
g.map_dataframe(plot_val_recon_loss)

# Set the labels for x-axis and y-axis
g.set_axis_labels('Epochs', 'Reconstruction Loss')

g.set_titles(col_template="beta = {col_name}", row_template="h_dim = {row_name}")

g.add_legend()

g.savefig(save_path + 'val_recon_loss_v5.png')

#%% Visualizing the Results for the KL Loss
save_path = 'outputs/Images/CVAnalysis/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


def plot_val_recon_loss(data, **kwargs):
    val_kl_loss = data['Validation KL Loss History'].values[0]
    e = len(val_kl_loss)
    epoch_list = list(range(1, e + 1))
    plt.plot(epoch_list, val_kl_loss, **kwargs)


# Create a FacetGrid with latent dimensions as the column variable
g = sns.FacetGrid(results, row='Hidden Dimensions', col='Latent Dimensions', hue='Beta', margin_titles=True)

# Map the function to each facet (latent dimension)
g.map_dataframe(plot_val_recon_loss)

# Set the labels for x-axis and y-axis
g.set_axis_labels('Epochs', 'KL Loss')

g.set_titles(col_template="beta = {col_name}", row_template="h_dim = {row_name}")

g.add_legend()

g.savefig(save_path + 'val_kl_loss_v6.png')
