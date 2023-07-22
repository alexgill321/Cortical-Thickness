""" Script to perform cross validation on the VAE model for the thickness data.

This script also contains some general visualizations of the results of the cross validation, with the intention of
 understanding the distribution of the results over the tested hyperparameters.
"""
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
save_path = cur + '/../outputs/models/vae/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#%% Load Data
filepath = os.path.join(cur, '../data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
train_data, val_data, test_data = generate_data_thickness_only(filepath)
val_batch_size = val_data.cardinality().numpy()
val_data_batched = val_data.batch(val_batch_size)
input_dim = train_data.element_spec[0].shape[0]

#%% Defining the Cross Validation
latent_dims = [2, 3, 4, 5, 7, 10, 15, 20]
beta = [0.001, 0.005, 0.0005]
hidden_dims = [[25, 25], [50, 25], [50, 50], [100, 50], [75, 50], [150, 100], [150, 100, 50]]
dropout = [0.2]
epochs = 200

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%% Running the Cross Validation
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%% Saving the Results
with open('../outputs/CrossVal/cv_thickness.pkl', 'wb') as file:
    pickle.dump(results, file)

#%% Visualizing Results
with open('../outputs/CrossVal/cv_thickness.pkl', 'rb') as file:
    results = pickle.load(file)

df = pd.DataFrame(columns=['Hidden Dimensions', 'Latent Dimensions', 'Beta', 'Dropout', 'Activation', 'Initializer',
                           'Total Loss', 'Reconstruction Loss', 'KL Loss', 'R2'])

#%% Extracting the Results
for model in results:
    (params, result) = model
    vae = load_or_train_model(save_path, params, train_data, epochs)
    _, r2, _, _ = vae.evaluate(val_data_batched)
    # Extract the parameters
    hidden_dims = params['encoder']['hidden_dim']
    latent_dim = params['encoder']['latent_dim']
    beta = params['vae']['beta']
    dropout_rate = params['encoder']['dropout_rate']
    activation = params['encoder']['activation']
    initializer = params['encoder']['initializer']

    # Extract the results
    total_loss = result['total_loss']
    recon_loss = result['recon_loss']
    kl_loss = result['kl_loss']

    # Create a new row in the DataFrame with the extracted values
    new_row = {'Hidden Dimensions': hidden_dims, 'Latent Dimensions': latent_dim, 'Beta': beta,
               'Dropout': dropout_rate, 'Activation': activation, 'Initializer': initializer,
               'Total Loss': total_loss, 'Reconstruction Loss': recon_loss, 'KL Loss': kl_loss, 'R2': r2}

    # Append the new row to the DataFrame
    df.loc[len(df)] = new_row
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


save_path = '../../outputs/Images/CVAnalysis/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%% Visualizing the Results for the Reconstruction Loss

# Convert the DataFrame Hidden Dimensions column to a string
df['Hidden Dimensions'] = df['Hidden Dimensions'].astype(str)

# Determine the overall minimum and maximum reconstruction loss values
min_loss = df['Reconstruction Loss'].min()
max_loss = df['Reconstruction Loss'].max()

# Reshape the DataFrame using pivot
# Create a FacetGrid with hidden dimensions as the column variable
g = sns.FacetGrid(df, col='Hidden Dimensions')

# Map the function to each facet (hidden dimension)
g.map_dataframe(create_heatmap_recon)

# Set the labels for x-axis and y-axis
g.set_axis_labels('Latent Dimensions', 'Beta')

# Save the plot
plt.savefig(save_path + 'heatmap_recon_loss.png')
#%% Visualizing the Results for the KL Loss

# Determine the overall minimum and maximum KL loss values
min_loss = df['KL Loss'].min()
max_loss = df['KL Loss'].max()

print(min_loss, max_loss)

# Create a FacetGrid with hidden dimensions as the column variable
g = sns.FacetGrid(df, col='Hidden Dimensions')

# Map the function to each facet (hidden dimension)
g.map_dataframe(create_heatmap_kl)

# Set the labels for x-axis and y-axis
g.set_axis_labels('Latent Dimensions', 'Beta')

# Save the plot
plt.savefig(save_path + 'heatmap_kl_loss.png')

#%% Visualizing the Results for the R2

# Determine the overall minimum and maximum KL loss values
min_loss = df['R2'].min()
max_loss = df['R2'].max()

print(min_loss, max_loss)

# Create a FacetGrid with hidden dimensions as the column variable
g = sns.FacetGrid(df, col='Hidden Dimensions')

# Map the function to each facet (hidden dimension)
g.map_dataframe(create_heatmap_r2)

# Set the labels for x-axis and y-axis
g.set_axis_labels('Latent Dimensions', 'Beta')

# Save the plot
plt.savefig(save_path + 'heatmap_r2.png')