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
from vaeModelAnalyzer import VAEModelAnalyzer

cur = os.getcwd()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
save_path = cur + '/outputs/models/vae_gn/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#%% Load Data
filepath = os.path.join(cur, 'data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
train_data, val_data, test_data, feat_labels = generate_data_thickness_only(filepath, normalize=2)
val_batch_size = val_data.cardinality().numpy()
val_data_batched = val_data.batch(val_batch_size)
input_dim = train_data.element_spec[0].shape[0]

#%% Defining the Cross Validation
latent_dims = [2, 3, 4, 5, 6]
beta = [1e-4]
hidden_dims = [[300, 150]]
dropout = [0.1]
lr = [1e-4]
epochs = 400

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%% Running the Cross Validation
results = cv.cross_validate_df(train_data, epochs=epochs, verbose=0)
with open('outputs/CrossVal/cv_dim_eval_v2.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%
with open('outputs/CrossVal/cv_thickness_global_norm.pkl', 'rb') as file:
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

#%%
results.drop_duplicates(inplace=True, subset=['Hidden Dimensions', 'Latent Dimensions', 'Beta'])
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
plt.savefig(save_path + 'heatmap_recon_loss_gn.png')

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
plt.savefig(save_path + 'heatmap_kl_loss_gn.png')

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
plt.savefig(save_path + 'heatmap_r2_gn.png')

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
plt.savefig(save_path + 'heatmap_total_loss_gn.png')

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

g.savefig(save_path + 'val_recon_loss_gn.png')

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

g.savefig(save_path + 'val_kl_loss_gn.png')

#%% Visualization for analysis comparing norm methods
with open('outputs/CrossVal/cv_thickness_global_norm.pkl', 'rb') as file:
    gn_res = pickle.load(file)
    gn_res['dataset_id'] = 'global_norm'
    gn_res['Hidden Dimensions'] = gn_res['Hidden Dimensions'].astype(str)
    gn_res.drop_duplicates(inplace=True, subset=['Hidden Dimensions', 'Latent Dimensions', 'Beta'])

with open('outputs/CrossVal/cv_thickness_norm.pkl', 'rb') as file:
    sn_res = pickle.load(file)
    sn_res['dataset_id'] = 'standard_norm'
    sn_res['Hidden Dimensions'] = sn_res['Hidden Dimensions'].astype(str)

with open('outputs/CrossVal/cv_thickness_v5.pkl', 'rb') as file:
    nn_res = pickle.load(file)
    nn_res['dataset_id'] = 'no_norm'
    nn_res['Hidden Dimensions'] = nn_res['Hidden Dimensions'].astype(str)

combined_df = pd.concat([gn_res, sn_res, nn_res], ignore_index=True)

#%%

min_loss = combined_df['R2'].min()
max_loss = combined_df['R2'].max()

# global norm r2
g = sns.FacetGrid(gn_res, col='Hidden Dimensions')

g.map_dataframe(create_heatmap_r2)

g.set_axis_labels('Latent Dimensions', 'Beta')

g.set_titles(col_template="H_dim = {col_name}", fontsize=20)

plt.savefig(save_path + 'heatmap_r2_gn.png')

# standard norm r2
g = sns.FacetGrid(sn_res, col='Hidden Dimensions')

g.map_dataframe(create_heatmap_r2)

g.set_axis_labels('Latent Dimensions', 'Beta')

g.set_titles(col_template="H_dim = {col_name}", fontsize=20)

plt.savefig(save_path + 'heatmap_r2_sn.png')

# no norm r2
g = sns.FacetGrid(nn_res, col='Hidden Dimensions')

g.map_dataframe(create_heatmap_r2)

g.set_axis_labels('Latent Dimensions', 'Beta')

g.set_titles(col_template="H_dim = {col_name}", fontsize=25)

plt.savefig(save_path + 'heatmap_r2_nn.png')

#%%
sns.boxplot(data=combined_df, x='R2', y='dataset_id')
plt.xlabel('R2 Scores')
plt.ylabel('Normalization Method')
plt.yticks(rotation=50)
plt.tight_layout()
plt.savefig(save_path + 'boxplot_r2.png')

#%% Get top model for each normalization method
idx = combined_df.groupby('dataset_id')['Total Loss'].idxmax()

# Retrieve the rows corresponding to the indexes
result = combined_df.loc[idx]

#%% Visualize the top models
save_path = 'outputs/analysis/'

# global norm r2
gn_data = result.where(result['dataset_id'] == 'global_norm').dropna()

params = gn_data['Parameters'].values[0]
gn_vae = load_or_train_model(params=params, train_data=train_data, epochs=400, verbose=1)
#%%
z_dim = int(gn_data['Latent Dimensions'].values[0])
analyzer = VAEModelAnalyzer(gn_vae, next(iter(val_data_batched)), z_dim, feat_labels, hist=None)
save_file = os.path.join(save_path, 'gn_top_total_model')
if not os.path.exists(save_file):
    os.makedirs(save_file)
analyzer.full_stack(save_file)

#%% Same for standard norm
train_data, val_data, test_data, feat_labels = generate_data_thickness_only(filepath, normalize=1)
val_batch_size = val_data.cardinality().numpy()
val_data_batched = val_data.batch(val_batch_size)
input_dim = train_data.element_spec[0].shape[0]

sn_data = result.where(result['dataset_id'] == 'standard_norm').dropna()

params = sn_data['Parameters'].values[0]
sn_vae = load_or_train_model(params=params, train_data=train_data, epochs=400, verbose=1)
#%%
z_dim = int(sn_data['Latent Dimensions'].values[0])
analyzer = VAEModelAnalyzer(sn_vae, next(iter(val_data_batched)), z_dim, feat_labels, hist=None)
save_file = os.path.join(save_path, 'sn_top_total_model')
if not os.path.exists(save_file):
    os.makedirs(save_file)
analyzer.full_stack(save_file)

#%% Same for no norm
train_data, val_data, test_data, feat_labels = generate_data_thickness_only(filepath, normalize=0)
val_batch_size = val_data.cardinality().numpy()
val_data_batched = val_data.batch(val_batch_size)
input_dim = train_data.element_spec[0].shape[0]

nn_data = result.where(result['dataset_id'] == 'no_norm').dropna()

params = nn_data['Parameters'].values[0]
nn_vae = load_or_train_model(params=params, train_data=train_data, epochs=400, verbose=1)

#%%
z_dim = int(nn_data['Latent Dimensions'].values[0])
analyzer = VAEModelAnalyzer(nn_vae, next(iter(val_data_batched)), z_dim, feat_labels, hist=None)
save_file = os.path.join(save_path, 'nn_top_total_model')
if not os.path.exists(save_file):
    os.makedirs(save_file)
analyzer.full_stack(save_file)

#%%
z_3 = gn_res.where(gn_res['Latent Dimensions'] == 3).dropna()
top = z_3.sort_values(by='R2').tail(5)