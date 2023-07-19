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
beta = [1e-9]
hidden_dims = [[100, 100], [150, 150], [200, 200], [300, 300], [400, 400], [300, 200]]
epochs = 150

dropout = [0.3, 0.4]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 3, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_no_kl_latent_10.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%

latent_dims = [15]
beta = [1e-9]
hidden_dims = [[100, 100], [150, 150], [200, 200], [300, 300], [400, 400], [300, 200]]
epochs = 150

dropout = [0.3, 0.4]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 3, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_no_kl_latent_15.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%

latent_dims = [20]
beta = [1e-9]
hidden_dims = [[100, 100], [150, 150], [200, 200], [300, 300], [400, 400], [300, 200]]
epochs = 150

dropout = [0.3, 0.4]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 3, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_no_kl_latent_20.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%

latent_dims = [25]
beta = [1e-9]
hidden_dims = [[100, 100], [150, 150], [200, 200], [300, 300], [400, 400], [300, 200]]
epochs = 150

dropout = [0.3, 0.4]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 3, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_no_kl_latent_25.pkl', 'wb') as file:
    pickle.dump(results, file)

#%%

latent_dims = [30]
beta = [1e-9]
hidden_dims = [[100, 100], [150, 150], [200, 200], [300, 300], [400, 400], [300, 200]]
epochs = 150

dropout = [0.3, 0.4]

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 3, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_no_kl_latent_30.pkl', 'wb') as file:
    pickle.dump(results, file)