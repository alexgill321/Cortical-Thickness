import pickle as pkl
import os
from modelUtils.vae_utils import load_or_train_model
from utils import data_validation, generate_feature_names
import vis_utils as vu

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cur = os.getcwd()
cv_res = []

# %% Load data
filepath = os.path.join(cur, 'data/cleaned_data/megasample_cleaned.csv')
feat_labels = generate_feature_names(filepath)
train_data, val_data, test_data = data_validation(filepath, validation_split=0.2)
val_batch_size = val_data.cardinality().numpy()
val_data_batched = val_data.batch(val_batch_size)
data = next(iter(val_data_batched))
input_dim = train_data.element_spec[0].shape[0]
path = os.path.join(cur, 'outputs/models/vae/')

# %% Control Result Generation
P1 = False
P2 = False
P3 = False
P4 = True
P5 = True
# %% load cross validation results
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

# %% Generate top reconstruction loss models for refined cv (top 5 and top 1) and total loss (top 1)
trained_top_models_cv_res = []
total_top_models_cv_res = []
trained_top_5_cv_res = []

for i in range(len(cv_res)):
    top_5 = []
    top_model = None
    total_top_model = None
    for model in cv_res[i]:
        (params, res) = model
        if top_model is None:
            top_model = model
        elif res['recon_loss'] < top_model[1]['recon_loss']:
            top_model = model
        if total_top_model is None:
            total_top_model = model
        elif res['total_loss'] < total_top_model[1]['total_loss']:
            total_top_model = model
        if len(top_5) < 5:
            top_5.append(model)
        else:
            top_5.sort(key=lambda x: x[1]['recon_loss'])
            if res['recon_loss'] < top_5[-1][1]['recon_loss']:
                top_5.pop()
                top_5.append(model)
    trained_top_5 = []
    for model in top_5:
        (params, _) = model
        vae = load_or_train_model(path, params, train_data, 300)
        trained_top_5.append(vae)
    trained_top_5_cv_res.append(trained_top_5)

    (params, _) = top_model
    vae = load_or_train_model(path, params, train_data, 300)
    trained_top_models_cv_res.append(vae)

    (params, _) = total_top_model
    vae = load_or_train_model(path, params, train_data, 50)
    total_top_models_cv_res.append(vae)

# %%
cv_og = []
with open(cur + '/outputs/CrossVal/cv_latent_10.pkl', 'rb') as file:
    cv_og.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_latent_15.pkl', 'rb') as file:
    cv_og.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_latent_20.pkl', 'rb') as file:
    cv_og.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_latent_25.pkl', 'rb') as file:
    cv_og.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_latent_30.pkl', 'rb') as file:
    cv_og.append(pkl.load(file))

# %% Generate top reconstruction loss models for original cv (top 5 and top 1) and total loss (top 1)
trained_top_models_cv_og = []
total_top_models_cv_og = []
trained_top_5_cv_og = []

for i in range(len(cv_og)):
    top_5 = []
    top_model = None
    total_top_model = None
    for model in cv_og[i]:
        (params, res) = model
        if top_model is None:
            top_model = model
        elif res['recon_loss'] < top_model[1]['recon_loss']:
            top_model = model
        if total_top_model is None:
            total_top_model = model
        elif res['total_loss'] < total_top_model[1]['total_loss']:
            total_top_model = model
        if len(top_5) < 5:
            top_5.append(model)
        else:
            top_5.sort(key=lambda x: x[1]['recon_loss'])
            if res['recon_loss'] < top_5[-1][1]['recon_loss']:
                top_5.pop()
                top_5.append(model)
    trained_top_5 = []
    for model in top_5:
        (params, res) = model
        vae = load_or_train_model(path, params, train_data, 150)
        trained_top_5.append(vae)
    trained_top_5_cv_og.append(trained_top_5)

    (params, _) = top_model
    vae = load_or_train_model(path, params, train_data, 150)
    trained_top_models_cv_og.append(vae)

    (params, _) = total_top_model
    vae = load_or_train_model(path, params, train_data, 150)
    total_top_models_cv_og.append(vae)

# %%
cv_no_kl = []
with open(cur + '/outputs/CrossVal/cv_no_kl_latent_10.pkl', 'rb') as file:
    cv_no_kl.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_no_kl_latent_15.pkl', 'rb') as file:
    cv_no_kl.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_no_kl_latent_20.pkl', 'rb') as file:
    cv_no_kl.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_no_kl_latent_25.pkl', 'rb') as file:
    cv_no_kl.append(pkl.load(file))
with open(cur + '/outputs/CrossVal/cv_no_kl_latent_30.pkl', 'rb') as file:
    cv_no_kl.append(pkl.load(file))
# %% Generate top reconstruction loss models for no kl cv (top 1) and total loss (top 1)
trained_top_models_cv_no_kl = []
total_top_models_cv_no_kl = []
for i in range(len(cv_no_kl)):
    top_model = None
    total_top_model = None
    for model in cv_no_kl[i]:
        if top_model is None:
            top_model = model
        elif model[1]['recon_loss'] < top_model[1]['recon_loss']:
            top_model = model
        if total_top_model is None:
            total_top_model = model
        elif model[1]['total_loss'] < total_top_model[1]['total_loss']:
            total_top_model = model

    (params, _) = top_model
    vae = load_or_train_model(path, params, train_data, 100)
    trained_top_models_cv_no_kl.append(vae)

    (params, _) = total_top_model
    vae = load_or_train_model(path, params, train_data, 100)
    total_top_models_cv_no_kl.append(vae)

# %% P1 Determine whether the latent space converges on a common representation, or if it changes with parameters
if P1:
    for i in range(len(cv_res)):
        print(f"Visualization of Latent dim: {labels[i]}")
        save_dir = cur + '/outputs/Images/latent_space/P1'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        savefile = save_dir + '/latent_space_visualization_' + labels[i] + '.png'
        vu.visualize_latent_space_multiple(trained_top_5_cv_res[i], data,
                                           labels=['vae1', 'vae2', 'vae3', 'vae4', 'vae5'], savefile=savefile)
        savefile = save_dir + '/latent_dim_visualization_' + labels[i] + '.png'
        vu.plot_latent_dimensions_multiple(trained_top_5_cv_res[i], data,
                                           labels=['vae1', 'vae2', 'vae3', 'vae4', 'vae5'], z_dim=int(labels[i]),
                                           savefile=savefile)
        savefile = save_dir + '/og_latent_space_visualization_' + labels[i] + '.png'
        vu.visualize_latent_space_multiple(trained_top_5_cv_og[i], data,
                                           labels=['vae1', 'vae2', 'vae3', 'vae4', 'vae5'], savefile=savefile)
        savefile = save_dir + '/og_latent_dim_visualization_' + labels[i] + '.png'
        vu.plot_latent_dimensions_multiple(trained_top_5_cv_og[i], data,
                                           labels=['vae1', 'vae2', 'vae3', 'vae4', 'vae5'], z_dim=int(labels[i]),
                                           savefile=savefile)

# %%
if P1:
    for i in range(len(cv_res)):
        print(f"Latent Dimension: {labels[i]}")
        save_dir = cur + '/outputs/Images/latent_space/P1'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # File paths for the images
        savefile = save_dir + '/No_kl_latent_space_visualization_' + labels[i] + '.png'
        savefile_1 = save_dir + '/no_kl_latent_clustering_' + labels[i] + '.png'
        savefile_2 = save_dir + '/clustered_labels_latent_space_visualization_' + labels[i] + '.png'
        savefile_3 = save_dir + '/og_clustered_labels_latent_space_visualization_' + labels[i] + '.png'
        savefile_4 = save_dir + '/no_kl_latent_dim_visualization_' + labels[i] + '.png'

        vu.visualize_latent_space(trained_top_models_cv_no_kl[i], data, savefile=savefile)
        cluster_labels = vu.latent_clustering(trained_top_models_cv_no_kl[i], data, num_clusters=5,
                                              savefile=savefile_1)
        vu.visualize_latent_space(trained_top_models_cv_res[i], data, labels=cluster_labels,
                                  savefile=savefile_2)
        vu.visualize_latent_space(trained_top_models_cv_og[i], data, labels=cluster_labels,
                                  savefile=savefile_3)
        vu.plot_latent_dimensions(trained_top_models_cv_no_kl[i], data, z_dim=int(labels[i]), savefile=savefile_4)

        # Plot the images
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # create a subplot with 1 row and 3 columns
        img2 = mpimg.imread(savefile_2)  # read the saved image
        axs[0].imshow(img2)  # show the image in the second column
        axs[0].axis('off')  # hide axes for this subplot
        axs[0].set_title('Clustered New CV')
        img3 = mpimg.imread(savefile_3)  # read the saved image
        axs[1].imshow(img3)  # show the image in the third column
        axs[1].axis('off')  # hide axes for this subplot
        axs[1].set_title('Clustered Original CV')
        img1 = mpimg.imread(savefile_1)  # read the saved image
        axs[2].imshow(img1)  # show the image in the first column
        axs[2].axis('off')  # hide axes for this subplot
        axs[2].set_title('Clustered No KL')
        plt.tight_layout()
        plt.savefig(save_dir + '/Structured_latent_space_visualization_' + labels[i] + '.png')
        plt.close()

# %% P2 Determine whether patients with similar clinical characteristics have similar latent representations
num_clusters = 30
kmeans = KMeans(num_clusters, n_init='auto', random_state=42)
cluster_labels = kmeans.fit_predict(data[0])

# Calculate silhouette scores for all samples
silhouette_vals = silhouette_samples(data[0], cluster_labels)

# Assign silhouette score to each cluster (average of all scores in the cluster)
silhouette_scores = []
for i in range(num_clusters):
    score = np.mean(silhouette_vals[cluster_labels == i])
    silhouette_scores.append((i, score))

# Sort clusters by score and select top 5
top_clusters = sorted(silhouette_scores, key=lambda x: x[1], reverse=True)[:5]

# Get the indices of the validation samples in each top cluster
top_cluster_indices = []
for idx, _ in top_clusters:
    top_indexes = np.where(cluster_labels == idx)[0]
    top_cluster_indices.append(top_indexes)
    print(f"Cluster {idx} has {len(top_indexes)} samples")

# %%
if P2:
    for i in range(len(cv_res)):
        print(f"Top 5 clusters for {labels[i]} latent dimensions")
        save_dir = cur + '/outputs/Images/latent_space/P2'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # File paths for the images
        savefile_1 = save_dir + '/top_5_clusters_visualization_' + labels[i] + '.png'
        savefile_2 = save_dir + '/no_kl_top_5_clusters_visualization_' + labels[i] + '.png'
        savefile_3 = save_dir + '/og_top_5_clusters_visualization_' + labels[i] + '.png'

        vu.visualize_top_clusters(trained_top_models_cv_res[i], data, top_cluster_indices, savefile=savefile_1)

        vu.visualize_top_clusters(trained_top_models_cv_no_kl[i], data, top_cluster_indices, savefile=savefile_2)

        vu.visualize_top_clusters(trained_top_models_cv_og[i], data, top_cluster_indices, savefile=savefile_3)

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))  # create a subplot with 1 row and 3 columns
        img1 = mpimg.imread(savefile_1)  # read the saved image
        axs[0].imshow(img1)  # show the image in the first column
        axs[0].axis('off')  # hide axes for this subplot
        axs[0].set_title('New CV Top 5 Clusters')
        img3 = mpimg.imread(savefile_3)  # read the saved image
        axs[1].imshow(img3)  # show the image in the third column
        axs[1].axis('off')  # hide axes for this subplot
        axs[1].set_title('Original CV Top 5 Clusters')
        img2 = mpimg.imread(savefile_2)  # read the saved image
        axs[2].imshow(img2)  # show the image in the second column
        axs[2].axis('off')  # hide axes for this subplot
        axs[2].set_title('No KL Top 5 Clusters')

        plt.tight_layout()
        plt.savefig(save_dir + '/Structured_top_5_clusters_visualization_' + labels[i] + '.png')
        plt.close()

# %% P3 Determine how influential each of the latent dimensions are on the data reconstruction
if P3:
    for i in range(len(cv_res)):
        save_dir = cur + '/outputs/Images/latent_space/P3'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # File paths for the images
        savefile_1 = save_dir + '/latent_entanglement_' + labels[i] + '.png'
        savefile_2 = save_dir + '/no_kl_latent_entanglement_' + labels[i] + '.png'
        savefile_3 = save_dir + '/og_latent_entanglement_' + labels[i] + '.png'

        vu.visualize_latent_interpolation(trained_top_models_cv_res[i], data, feat_labels=feat_labels,
                                          z_dim=int(labels[i]), savefile=savefile_1)
        vu.visualize_latent_interpolation(trained_top_models_cv_no_kl[i], data, feat_labels=feat_labels,
                                          z_dim=int(labels[i]), savefile=savefile_2)
        vu.visualize_latent_interpolation(trained_top_models_cv_og[i], data, feat_labels=feat_labels,
                                          z_dim=int(labels[i]), savefile=savefile_3)

        savefile_1 = save_dir + '/latent_dim_influence_' + labels[i] + '.png'
        savefile_2 = save_dir + '/no_kl_latent_dim_influence_' + labels[i] + '.png'
        savefile_3 = save_dir + '/og_latent_dim_influence_' + labels[i] + '.png'

        vu.visualize_latent_influence(trained_top_models_cv_res[i], data, z_dim=int(labels[i]), savefile=savefile_1)
        vu.visualize_latent_influence(trained_top_models_cv_no_kl[i], data, z_dim=int(labels[i]), savefile=savefile_2)
        vu.visualize_latent_influence(trained_top_models_cv_og[i], data, z_dim=int(labels[i]), savefile=savefile_3)

# %% P4 Determine how accurate the data reconstructions are
if P4:
    # evaluate reconstructions of validation data on the top models using R squared metric
    for i in range(len(cv_res)):
        print(f"R squared for {labels[i]} latent dimensions")
        print(f"Original R squared (reconstruction loss minimization)")
        _, r2_og, _, _ = trained_top_models_cv_og[i].evaluate(val_data_batched)
        print(f"No KL R squared (reconstruction loss minimization)")
        _, r2_no_kl, _, _ = trained_top_models_cv_no_kl[i].evaluate(val_data_batched)
        print(f"New R squared (reconstruction loss minimization)")
        _, r2_res, _, _ = trained_top_models_cv_res[i].evaluate(val_data_batched)

        print(f"Original R squared (total loss minimization)")
        _, total_r2_og, _, _ = total_top_models_cv_og[i].evaluate(val_data_batched)
        print(f"No KL R squared (total loss minimization)")
        _, total_r2_no_kl, _, _ = total_top_models_cv_no_kl[i].evaluate(val_data_batched)
        print(f"New R squared (total loss minimization)")
        _, total_r2_res, _, _ = total_top_models_cv_res[i].evaluate(val_data_batched)

# %% P4
if P4:
    for i in range(len(cv_res)):
        save_dir = cur + '/outputs/Images/latent_space/P4'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # File paths for the images
        savefile_1 = save_dir + '/reconstruction_' + labels[i] + '.png'
        savefile_2 = save_dir + '/no_kl_reconstruction_' + labels[i] + '.png'
        savefile_3 = save_dir + '/og_reconstruction_' + labels[i] + '.png'

        vu.visualize_errors_hist(trained_top_models_cv_res[i], data, savefile=savefile_1)
        vu.visualize_errors_hist(trained_top_models_cv_no_kl[i], data, savefile=savefile_2)
        vu.visualize_errors_hist(trained_top_models_cv_og[i], data, savefile=savefile_3)

        save_dir = cur + '/outputs/Data/P4'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # File paths for the datasets
        savefile_1 = save_dir + '/reconstruction_errors_' + labels[i] + '.csv'
        savefile_2 = save_dir + '/no_kl_reconstruction_errors_' + labels[i] + '.csv'
        savefile_3 = save_dir + '/og_reconstruction_errors_' + labels[i] + '.csv'
        vu.calc_feature_errors(trained_top_models_cv_res[i], data, feat_labels=feat_labels, savefile=savefile_1)
        vu.calc_feature_errors(trained_top_models_cv_no_kl[i], data, feat_labels=feat_labels, savefile=savefile_2)
        vu.calc_feature_errors(trained_top_models_cv_og[i], data, feat_labels=feat_labels, savefile=savefile_3)


# %% P5 Determine whether clusters in the latent space correspond to clusters in the data space

# %% P6 Examine reconstruction errors and distributions for particular brain regions
