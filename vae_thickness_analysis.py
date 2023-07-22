import pickle as pkl
import os
from utils import generate_data_thickness_only
from modelUtils.vae_utils import load_or_train_model, get_filename_from_params
from vaeModelAnalyzer import VAEModelAnalyzer

cur = os.getcwd()
#%% Load Data
filepath = os.path.join(cur, 'data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
train_data, val_data, test_data, feat_labels = generate_data_thickness_only(filepath)
val_batch_size = val_data.cardinality().numpy()
val_data_batched = val_data.batch(val_batch_size)
data = next(iter(val_data_batched))

with open(os.path.join(cur, 'outputs/CrossVal/cv_thickness_v3.pkl'), 'rb') as file:
    cv_results = pkl.load(file)

#%% Retrieve top 5 models for Reconstruction Loss
top_5_recon = cv_results.sort_values(by='Reconstruction Loss').head(5)
print(top_5_recon)
#%%
for i, row in top_5_recon.iterrows():
    model = load_or_train_model(os.path.join(cur, 'outputs/models/vae'), row['Parameters'], train_data, epochs=300)
    analyzer = VAEModelAnalyzer(model, data, row['Latent Dimensions'], feat_labels)
    save_path = os.path.join(cur, 'outputs/analysis/recon', get_filename_from_params(row['Parameters'], 300))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    analyzer.full_stack(save_path)

#%% Retrieve top 5 models for Total Loss
top_5_total = cv_results.sort_values(by='Total Loss').head(5)
print(top_5_total)
#%%
for i, row in top_5_total.iterrows():
    model = load_or_train_model(os.path.join(cur, 'outputs/models/vae'), row['Parameters'], train_data, epochs=300)
    analyzer = VAEModelAnalyzer(model, data, row['Latent Dimensions'], feat_labels)
    save_path = os.path.join(cur, 'outputs/analysis/total', get_filename_from_params(row['Parameters'], 300))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    analyzer.full_stack(save_path)
