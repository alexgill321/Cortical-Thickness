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
save_path = cur + '/outputs/models/vae/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

#%% Load Data
filepath = os.path.join(cur, 'data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
train_data, val_data, test_data = generate_data_thickness_only(filepath)
val_batch_size = val_data.cardinality().numpy()
val_data_batched = val_data.batch(val_batch_size)
input_dim = train_data.element_spec[0].shape[0]

#%% Defining the Cross Validation
latent_dims = [3, 4, 5, 10]
beta = [0.001, 0.0006, 0.0003, 0.0001]
hidden_dims = [[150, 100], [200, 150], [150, 150], [200, 200], [200, 100], [300, 150]]
dropout = [0.2]
epochs = 300

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%% Running the Cross Validation
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%% Saving the Results
with open('/outputs/CrossVal/cv_thickness_v2.pkl', 'wb') as file:
    pickle.dump(results, file)