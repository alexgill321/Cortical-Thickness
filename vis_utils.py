from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.patches as mpatches
import tensorflow as tf
from scipy import stats


def visualize_latent_space(vae, data, labels=None, savefile=None):

    # 1. Encode input samples
    z_mean, z_log_var, z, _ = vae(data)
    z = z.numpy()

    # 2. Apply t-SNE
    tsne = TSNE(random_state=42)
    z_2d = tsne.fit_transform(z)

    # 3. Create a scatter plot
    fig = plt.figure(figsize=(8, 6))
    if labels is None:
        sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], alpha=0.6)
    else:
        sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=labels, alpha=0.6,
                        palette=sns.color_palette('hsv', len(set(labels))))
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if savefile is not None:
        plt.savefig(savefile)
    plt.close()


def visualize_latent_space_multiple(vae_models, data, labels, savefile=None):

    z_results = []
    fig = plt.figure(figsize=(8, 6))
    for i in range(len(vae_models)):
        model = vae_models[i]
        z_mean, z_log_var, z, _ = model(data)
        z = z.numpy()
        tsne = TSNE(n_components=2, random_state=42)
        z_2d = tsne.fit_transform(z)
        plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6, label=labels[i])

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()

    if savefile is not None:
        plt.savefig(savefile)
    plt.close()


def latent_clustering(vae, data, num_clusters, savefile=None):
    # Assuming X is your high dimensional data.

    # 1. Encode input samples
    z_mean, z_log_var, z, _ = vae(data)
    z = z.numpy()

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
    kmeans.fit(z)
    labels = kmeans.labels_

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=0)
    z_2d = tsne.fit_transform(z)

    # Create a DataFrame that will be used for plotting
    df = pd.DataFrame(z_2d, columns=['Component 1', 'Component 2'])
    df['Cluster'] = labels

    # Plot the results using Seaborn
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Component 1', y='Component 2', hue='Cluster',
                    palette=sns.color_palette('hsv', num_clusters))

    if savefile is not None:
        plt.savefig(savefile)
    plt.close()
    return labels


def plot_latent_dimensions(vae, data, z_dim, savefile=None):

    n_cols = 3
    n_rows = int(np.ceil(z_dim / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axs = axs.flatten()

    z_mean, z_log_var, z, _ = vae(data)
    z = z.numpy()

    for i in range(z_dim):
        sns.histplot(z[:, i], ax=axs[i], kde=True, color='blue', alpha=0.6, bins=20)
        axs[i].set_title(f"Latent dimension {i+1}")

    for i in range(z_dim, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
    plt.close()


def plot_latent_dimensions_multiple(vae_models, data, z_dim, labels, savefile=None):

    n_cols = 3
    n_rows = int(np.ceil(z_dim / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axs = axs.flatten()

    # Store the latent vectors for each model
    zs = []
    for vae in vae_models:
        z_mean, z_log_var, z, _ = vae(data)
        zs.append(z.numpy())

    for i in range(z_dim):
        for j in range(len(vae_models)):
            sns.kdeplot(zs[j][:, i], ax=axs[i], label=labels[j])
        axs[i].set_title(f"Latent dimension {i+1}")
        axs[i].legend()

    for i in range(z_dim, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
    plt.close()


def visualize_top_clusters(vae, data, top_cluster_indices, savefile=None):

    # 1. Encode input samples
    z_mean, z_log_var, z, _ = vae(data)
    z = z.numpy()

    # 2. Apply t-SNE
    tsne = TSNE(random_state=42)
    z_2d = tsne.fit_transform(z)

    # 3. Create the labels for the clusters
    color_labels = np.zeros(z_2d.shape[0])
    for i in range(len(top_cluster_indices)):
        color_labels[top_cluster_indices[i]] = i + 1

    # 4. Create a scatter plot
    plt.figure(figsize=(8, 6))
    color_palette = ["gray"] + sns.color_palette('hsv', len(set(color_labels)) - 1)  # use "gray" for ungrouped data
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=color_labels, alpha=0.6, palette=color_palette, legend=False)

    # 5. Create legend
    patch_list = []
    for i in range(int(np.max(color_labels)) + 1):
        if i == 0:
            label = "un-clustered"
        else:
            label = f"cluster {i}"
        data_key = mpatches.Patch(color=color_palette[i], label=label)
        patch_list.append(data_key)
    plt.legend(handles=patch_list)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if savefile is not None:
        plt.savefig(savefile)
    plt.close()


def visualize_latent_influence(vae, data, z_dim, savefile=None):

    # 1. Encode input samples
    z_mean, z_log_var, z, original_recon = vae(data)
    z = z.numpy()

    # Prepare to collect mean errors for each dimension
    mean_errors = []

    # 2. For each dimension in the latent space
    for i in range(z_dim):
        # vary the i-th dimension
        z_altered = np.copy(z)
        original_mean = np.mean(z[:, i])
        increase = 1.5 * original_mean - original_mean
        z_altered[:, i] += increase

        # 3. Decode the altered latent space
        altered_recon = vae.decoder(z_altered)

        # 4. Compute the error between original and altered
        error = np.abs(original_recon - altered_recon)

        # Calculate and store the mean error
        mean_errors.append(np.mean(error))

    # Create a figure for visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Visualize the mean error for each dimension in a bar chart
    ax.bar(range(z_dim), mean_errors, tick_label=[f'Dim {i+1}' for i in range(z_dim)])
    ax.set_title('Mean Reconstruction Error for each Latent Dimension')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean Error')

    # save the figure if needed
    if savefile:
        plt.savefig(savefile)

    plt.tight_layout()
    plt.close()


def visualize_latent_interpolation(vae, data, z_dim, feat_labels, num_features=10, savefile=None):

    # 1. Encode input samples
    z_mean, z_log_var, z, original_recon = vae(data)
    z = z.numpy()

    # Determine the features to visualize
    # Here we randomly select num_features from the feature space
    num_total_features = original_recon.shape[-1]
    selected_features = np.random.choice(num_total_features, num_features, replace=False)

    # Prepare to collect mean errors for each dimension
    feature_errors = np.zeros((num_features, z_dim))

    # 2. For each dimension in the latent space
    for i in range(z_dim):
        # vary the i-th dimension
        z_altered = np.copy(z)
        original_mean = np.mean(z[:, i])
        increase = 1.5 * original_mean - original_mean
        z_altered[:, i] += increase

        # 3. Decode the altered latent space
        altered_recon = vae.decoder(z_altered)

        # 4. Compute the error between original and altered
        error = np.abs(original_recon - altered_recon)

        # Store the mean error for each selected feature
        for j, feature in enumerate(selected_features):
            feature_errors[j, i] = np.mean(error[:, feature])

    n_cols = 3
    n_rows = int(np.ceil(num_features / n_cols))

    # Create a figure for visualization
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(20, num_features * 2))
    axs = axs.flatten()
    # Visualize the mean error for each dimension in a pie chart
    for j in range(num_features):
        axs[j].bar(range(z_dim), feature_errors[j], tick_label=[f'{i+1}' for i in range(z_dim)])
        axs[j].set_title(f'{feat_labels[selected_features[j]]}')
        axs[j].set_xlabel('Latent Dimension')
        axs[j].set_ylabel('Mean Error')

    # Remove extra axes
    for i in range(num_features, len(axs)):
        axs[i].remove()

    if savefile:
        plt.savefig(savefile)

    plt.close()


def visualize_errors_hist(vae, data, savefile=None):
    """ Creates a Histogram of the mean reconstruction errors. """

    x, y = data
    x = tf.cast(x, dtype=tf.float32)
    # Generate the reconstruction
    z_mean, z_log_var, z, x_reconstruction = vae(data)

    # Calculate the error
    error = x - x_reconstruction

    # Calculate the mean error for each sample
    mean_errors = np.mean(error, axis=1)

    # Create a histogram of the mean errors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mean_errors, bins=50, density=True, alpha=0.6, color='g')

    # Fit a normal distribution to the data
    mu, std = stats.norm.fit(mean_errors)

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    ax.set_title(title)

    t_stat, p_val = stats.ttest_1samp(mean_errors, 0)
    if p_val < 0.05:
        ax.axvline(x=0, color='r', linestyle='--', label=f'p-value: {p_val:.2f}')
    else:
        ax.axvline(x=0, color='g', linestyle='--', label=f'p-value: {p_val:.2f}')

    ax.legend()
    ax.set_xlabel('Mean Error')
    ax.set_ylabel('Density')

    if savefile:
        plt.savefig(savefile)

    plt.close()


def calc_feature_errors(vae, data, feat_labels, savefile=None):
    """ Calculates the mean reconstruction error for each feature.

     Args:
        vae: The trained VAE model.
        data: The data to be used for the calculation.
        feat_labels: The labels for the features.
        savefile: The path to save the results to.

     Returns: A dataframe containing the mean reconstruction error for each feature.
     """
    x, y = data
    x = tf.cast(x, dtype=tf.float32)

    # Generate the reconstruction
    z_mean, z_log_var, z, x_reconstruction = vae(data)

    # Calculate the error
    error = x - x_reconstruction

    # Calculate the mean error for each of the features
    mean_errors = np.mean(error, axis=0)

    # Create a dataframe to store the results
    df = pd.DataFrame({'feature': feat_labels, 'mean_error': mean_errors})

    # Sort the dataframe by mean error
    df = df.sort_values(by='mean_error', ascending=False)

    if savefile:
        df.to_csv(savefile, index=False)

    return df


#%%
