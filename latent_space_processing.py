import pickle as pkl
import os
from modelUtils.vae_utils import load_vae, get_filename_from_params
from models.vae_models import VAE, create_vae_encoder, create_vae_decoder
from utils import data_validation
from vis_utils import visualize_latent_space_multiple, plot_latent_dimensions_multiple

cv_res = []

#%% Load data
cur = os.getcwd()
filepath = os.path.join(cur, 'outputs/megasample_cleaned.csv')
train_data, val_data, test_data = data_validation(filepath, validation_split=0.2)
input_dim = train_data.element_spec[0].shape[0]
#%% load cross validation results
with open(cur + '/outputs/CrossVal/cv_finetune_latent_10.pkl', 'rb') as file:
    cv_res.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_finetune_latent_15.pkl', 'rb') as file:
    cv_res.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_finetune_latent_20.pkl', 'rb') as file:
    cv_res.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_finetune_latent_25.pkl', 'rb') as file:
    cv_res.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_finetune_latent_30.pkl', 'rb') as file:
    cv_res.append(pkl.load(file))
labels = ['10', '15', '20', '25', '30']
#%% P1 Determine whether the latent space converges on a common representation, or if it changes with parameters
for i in range(len(cv_res)):
    top_5 = []
    for model in cv_res[i]:
        (params, res) = model
        if len(top_5) < 5:
            # If less than 5 models, simply append
            top_5.append(model)
        else:
            # If already 5 models, check if the current model is better than the worst in top_models_total
            top_5.sort(key=lambda x: x[1]['total_loss'])  # Sort by total_loss in ascending order
            if res['total_loss'] < top_5[-1][1]['total_loss']:
                # If current model is better, remove the worst and append the current
                top_5.pop()
                top_5.append(model)

    # Now we have the top 5 models for each latent dimension, we can compare the latent representations
    trained_models = []
    for model in top_5:
        (params, res) = model
        filename = get_filename_from_params(params, 300)
        filepath = os.path.join(cur, 'outputs/models/vae', filename)
        if os.path.exists(filepath):
            vae = load_vae(filepath)
            vae.compile()
            print(f"Loaded model from {filepath}")
        else:
            encoder = create_vae_encoder(input_dim=input_dim, **params['encoder'])
            decoder = create_vae_decoder(output_dim=input_dim, **params['decoder'])
            vae = VAE(encoder, decoder, **params['vae'])
            vae.compile()
            vae.fit(train_data.batch(128), epochs=300, verbose=0)
        trained_models.append(vae)
    save_dir = cur + '/outputs/Images/latent_space/P1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    savefile = save_dir + '/latent_space_visualization_' + labels[i] + '.png'
    visualize_latent_space_multiple(trained_models, val_data, labels=['vae1', 'vae2', 'vae3', 'vae4', 'vae5'],
                                    savefile=savefile)
    savefile = save_dir + '/latent_dim_visualization_' + labels[i] + '.png'
    plot_latent_dimensions_multiple(trained_models, val_data, labels=['vae1', 'vae2', 'vae3', 'vae4', 'vae5'],
                                    z_dim=int(labels[i]), savefile=savefile)


#%% P2 Determine whether patients with similar clinical characteristics have similar latent representations

#%% P3 Determine how influential each of the latent dimensions are on the data reconstruction

#%% P4 Determine how accurate the data reconstructions are

#%% P5 Determine whether clusters in the latent space correspond to clusters in the data space

#%% P6 Examine reconstruction errors and distributions for particular brain regions