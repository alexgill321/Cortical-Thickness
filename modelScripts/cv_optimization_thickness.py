from modelUtils.vae_utils import create_param_grid, VAECrossValidator
import os
import pickle
from utils import generate_data_thickness_only
import tensorflow as tf
import pandas as pd

cur = os.getcwd()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
save_path = cur + '/../outputs/models/vae/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#%%
filepath = os.path.join(cur, '../data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
train_data, val_data, test_data = generate_data_thickness_only(filepath)
input_dim = train_data.element_spec[0].shape[0]

#%%
latent_dims = [2, 3, 4, 5, 7, 10, 15, 20]
beta = [0.001, 0.005, 0.0005]
hidden_dims = [[25, 25], [50, 25], [50, 50], [100, 50], [75, 50], [150, 100], [150, 100, 50]]
dropout = [0.2]
epochs = 200

param_grid = create_param_grid(hidden_dims, latent_dims, dropout, ['relu'], ['glorot_uniform'], betas=beta)
cv = VAECrossValidator(param_grid, input_dim, 5, batch_size=128, save_path=save_path)

#%%
results = cv.cross_validate(train_data, epochs=epochs, verbose=0)

#%%
with open('../outputs/CrossVal/cv_thickness.pkl', 'wb') as file:
    pickle.dump(results, file)

#%% Visualizing Results
with open('../outputs/CrossVal/cv_thickness.pkl', 'rb') as file:
    results = pickle.load(file)

df = pd.DataFrame(columns=['Hidden Dimensions', 'Latent Dimensions', 'Beta', 'Dropout', 'Activation', 'Initializer',
                           'Total Loss', 'Reconstruction Loss', 'KL Loss'])
for model in results:
    (params, result) = model
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
               'Total Loss': total_loss, 'Reconstruction Loss': recon_loss, 'KL Loss': kl_loss}

    # Append the new row to the DataFrame
    df = df.append(new_row, ignore_index=True)

