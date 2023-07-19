""" Cross validates parameters with the goal of finding better latent space representations.

"""

from modelUtils.vae_utils import create_param_grid, VAECrossValidator
import os
from utils import data_validation
import pickle
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
save_path = os.getcwd() + '/outputs/models/vae/'
#%%
cur = os.getcwd()
filepath = os.path.join(cur, 'outputs/megasample_cleaned.csv')
train_data, val_data, test_data = data_validation(filepath)
input_dim = train_data.element_spec[0].shape[0]

#%%
# Cross Validation for 10 dimension latent space
latent_dims = [10]
beta = [1e-2, 1e-3, 1e-4]
hidden_dims = [[100, 50], [150, 100], [200, 100], [150, 100, 50], [200, 150, 100], [300, 200, 100], [150, 150, 100, 50],
               [200, 150, 100, 100], [300, 200, 200, 100], [200, 150, 100, 50]]
epochs = 300

dropout = [0.2, 0.3, 0.4]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_finetune_latent_10.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%
# Cross Validation for 15 dimension latent space
latent_dims = [15]
beta = [1e-2, 1e-3, 1e-4]
hidden_dims = [[100, 50], [150, 100], [200, 100], [150, 100, 50], [200, 150, 100], [300, 200, 100], [150, 150, 100, 50],
               [200, 150, 100, 100], [300, 200, 200, 100], [200, 150, 100, 50]]
epochs = 300

dropout = [0.2, 0.3, 0.4]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_finetune_latent_15.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%
# Cross Validation for 20 dimension latent space
latent_dims = [20]
beta = [1e-2, 1e-3, 1e-4]
hidden_dims = [[100, 50], [150, 100], [200, 100], [150, 100, 50], [200, 150, 100], [300, 200, 100], [150, 150, 100, 50],
               [200, 150, 100, 100], [300, 200, 200, 100], [200, 150, 100, 50]]
epochs = 300

dropout = [0.2, 0.3, 0.4]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_finetune_latent_20.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%
# Cross Validation for 25 dimension latent space
latent_dims = [25]
beta = [1e-2, 1e-3, 1e-4]
hidden_dims = [[100, 50], [150, 100], [200, 100], [150, 100, 50], [200, 150, 100], [300, 200, 100], [150, 150, 100, 50],
               [200, 150, 100, 100], [300, 200, 200, 100], [200, 150, 100, 50]]
epochs = 300

dropout = [0.2, 0.3, 0.4]
param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_finetune_latent_25.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%
# Cross Validation for 30 dimension latent space
latent_dims = [30]
beta = [1e-2, 1e-3, 1e-4]
hidden_dims = [[100, 50], [150, 100], [200, 100], [150, 100, 50], [200, 150, 100], [300, 200, 100],[300, 300, 300],
               [150, 150, 100, 50], [200, 150, 100, 100], [300, 200, 200, 100], [200, 150, 100, 50]]
epochs = 300
dropout = [0.2, 0.3, 0.4]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_finetune_latent_30.pkl', 'wb') as file:
    pickle.dump(results, file)
