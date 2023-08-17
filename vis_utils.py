from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_samples, mean_squared_error
import tensorflow as tf
from scipy import stats
from sklearn.linear_model import LinearRegression


def visualize_latent_space(vae, data, labels=None, savefile=None):
    """ Visualize the latent space of a VAE model

    This function encodes the input data using the vae model, then applies t-SNE to reduce the dimensionality of the
    latent space to 2D. The resulting 2D latent space is then plotted using a scatter plot.

    Args:
        vae: A trained VAE model
        data: A single batch of data to be used for the analysis
        labels (optional): A list of labels for the data
        savefile (optional): The path to the file to save the visualization to

    Returns: The figure
    """

    z_mean, z_log_var, z, _ = vae(data)
    z = z.numpy()

    tsne = TSNE(random_state=42)
    z_2d = tsne.fit_transform(z)

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
    return fig


def visualize_latent_space_multiple(vae_models, data, labels, savefile=None):
    """ Visualize the latent space of multiple VAE models

    This function encodes the input data using each of the vae models, then applies t-SNE to reduce the dimensionality
    of the latent space to 2D. The resulting 2D latent space is then plotted using a scatter plot.

    Args:
        vae_models: A list of trained VAE models
        data: A single batch of data to be used for the analysis
        labels: A list of labels for the data
        savefile (optional): The path to the file to save the visualization to

    Returns: The figure
    """
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
    return fig


def latent_clustering(vae, data, num_clusters, savefile=None):
    """ Cluster the latent space of a VAE model

    This function encodes the input data using the vae model, then applies KMeans clustering to the latent space. The
    resulting clusters are then plotted on the tsne reduced latent space using a scatter plot.

    Args:
        vae: A trained VAE model
        data: A single batch of data to be used for the analysis
        num_clusters: The number of clusters to use for the KMeans clustering
        savefile (optional): The path to the file to save the visualization to

    Returns: The figure and the cluster labels
    """
    z_mean, z_log_var, z, _ = vae(data)
    z = z.numpy()

    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
    kmeans.fit(z)
    labels = kmeans.labels_

    tsne = TSNE(n_components=2, random_state=0)
    z_2d = tsne.fit_transform(z)

    df = pd.DataFrame(z_2d, columns=['Component 1', 'Component 2'])
    df['Cluster'] = labels

    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Component 1', y='Component 2', hue='Cluster',
                    palette=sns.color_palette('hsv', num_clusters))

    if savefile is not None:
        plt.savefig(savefile)
    plt.close()
    return fig, labels


def plot_latent_dimensions(vae, data, z_dim, savefile=None):
    """ Plot the distribution of each dimension in the latent space

    This function encodes the input data using the vae model, then plots the distribution of each dimension in the
    latent space using a histogram.

    Args:
        vae: A trained VAE model
        data: A single batch of data to be used for the analysis
        z_dim: The dimensionality of the latent space
        savefile (optional): The path to the file to save the visualization to

    Returns: The figure
    """
    n_cols = 3
    n_rows = int(np.ceil(z_dim / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
    axs = axs.flatten()

    z_mean, z_log_var, z, _ = vae(data)
    z = z.numpy()

    for i in range(z_dim):
        sns.histplot(z[:, i], ax=axs[i], kde=True, color='blue', alpha=0.6, bins=20)
        mu, std = stats.norm.fit(z[:, i])
        t_stat, p_val = stats.ttest_1samp(z[:, i], 0)
        if p_val < 0.05:
            axs[i].axvline(x=mu, color='red', linestyle='--', linewidth=3, label=f"p={p_val:.2f}")
        else:
            axs[i].axvline(x=mu, color='green', linestyle='--', linewidth=3, label=f"p={p_val:.2f}")
        axs[i].legend(fontsize=20)
        axs[i].set_title(f"Dimension {i+1}: μ={mu:.2f}, σ={std:.2f}", fontsize=20)
        axs[i].axis('off')

    for i in range(z_dim, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
    plt.close()
    return fig


def plot_latent_dimensions_multiple(vae_models, data, z_dim, labels, savefile=None):
    """ Plot the distribution of each dimension in the latent space for multiple VAE models

    This function encodes the input data using each of the vae models, then plots the distribution of each dimension in
    the latent space using a histogram.

    Args:
        vae_models: A list of trained VAE models
        data: A single batch of data to be used for the analysis
        z_dim: The dimensionality of the latent space
        labels: A list of labels for the data
        savefile (optional): The path to the file to save the visualization to

    Returns: The figure
    """
    n_cols = 3
    n_rows = int(np.ceil(z_dim / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axs = axs.flatten()

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
    return fig


def visualize_top_clusters(vae, data, num_clusters, top_k, savefile=None):
    """ Visualize the top clusters on raw data in the latent space

    This function generates the top clusters on the raw input data in the high dimensional feature space using K means
    clustering. Then using silhouette scores it determines which clusters have the highest score.The input data is then
    encoded using the vae model, and tsne is used to reduce the dimensionality of the latent space to 2D. The resulting
    latent space is then plotted using a scatter plot, with the top clusters highlighted.

    The purpose of this visualization is to see if the clusters remain distinct in the latent space.

    Args:
        vae: A trained VAE model
        data: A single batch of data to be used for the analysis
        num_clusters: The number of clusters used for the KMeans clustering
        top_k: The number of top clusters to visualize
        savefile (optional): The path to the file to save the visualization to

    Returns: The figure
    """
    k_mean = KMeans(num_clusters, n_init='auto', random_state=42)
    cluster_labels = k_mean.fit_predict(data[0])

    silhouette_vals = silhouette_samples(data[0], cluster_labels)

    silhouette_scores = []
    for i in range(num_clusters):
        score = np.mean(silhouette_vals[cluster_labels == i])
        silhouette_scores.append((i, score))

    top_clusters = sorted(silhouette_scores, key=lambda x: x[1], reverse=True)[:top_k]
    top_cluster_indices = []

    for idx, _ in top_clusters:
        top_indexes = np.where(cluster_labels == idx)[0]
        top_cluster_indices.append(top_indexes)

    z_mean, z_log_var, z, _ = vae(data)
    z = z.numpy()

    tsne = TSNE(random_state=42)
    z_2d = tsne.fit_transform(z)

    color_labels = np.zeros(z_2d.shape[0])
    for i in range(len(top_cluster_indices)):
        color_labels[top_cluster_indices[i]] = i + 1

    fig = plt.figure(figsize=(8, 6))
    color_palette = ["gray"] + sns.color_palette('hsv', len(set(color_labels)) - 1)
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=color_labels, alpha=0.6, palette=color_palette, legend=False)

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
    return fig


def visualize_latent_influence(vae, data, z_dim, savefile=None):
    """ Visualize the influence of each dimension in the latent space

    This function encodes the input data using the vae model, then for each dimension in the latent space it alters the
    latent vector by increasing the value of that dimension by 50%. The altered latent vector is then decoded and the
    error between the original and altered input is calculated. The mean error for each dimension is then plotted.

    The purpose of this visualization is to see which dimensions in the latent space have the most influence on the
    output.

    Args:
        vae: A trained VAE model
        data: A single batch of data to be used for the analysis
        z_dim: The number of dimensions in the latent space
        savefile (optional): The path to the file to save the visualization to

    Returns: The figure
    """
    # 1. Encode input samples
    z_mean, z_log_var, z, original_recon = vae(data)
    z = z.numpy()

    # Prepare to collect mean errors for each dimension
    mean_errors = []

    # 2. For each dimension in the latent space
    for i in range(z_dim):
        # vary the i-th dimension
        z_altered = np.copy(z)
        z_altered[:, i] = 1.5 * z[:, i]

        # 3. Decode the altered latent space
        altered_recon = vae.decoder(z_altered)

        # 4. Compute the error between original and altered
        error = np.abs(original_recon - altered_recon)

        # Calculate and store the mean error
        mean_errors.append(np.mean(error))

    # Create a figure for visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Visualize the mean error for each dimension in a bar chart
    ax.bar(range(z_dim), mean_errors, tick_label=[f'{i+1}' for i in range(z_dim)])
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean Error')

    # save the figure if needed
    if savefile:
        plt.savefig(savefile)

    plt.tight_layout()
    plt.close()
    return fig


def visualize_latent_interpolation(vae, data, z_dim, feat_labels, num_features=3, savefile=None):
    """ Visualize the influence of each dimension in the latent space on individual features

    This function encodes the input data using the vae model, then for each dimension in the latent space it alters the
    latent vector by increasing the value of that dimension by 50%. The altered latent vector is then decoded and the
    error between the original and altered input is calculated. The mean error for each dimension is then plotted for
    each of the features being analyzed.

    The purpose of this visualization is to see which dimensions in the latent space have the most influence on the
    output of specific features.

    Args:
        vae: A trained VAE model
        data: A single batch of data to be used for the analysis
        z_dim: The number of dimensions in the latent space
        feat_labels: A list of the feature labels for the features being analyzed
        num_features: The number of features to analyze
        savefile (optional): The path to the file to save the visualization to

    Returns: The figure and a list of the selected features
    """
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
        increase = 1.2 * original_mean - original_mean
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
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(20, n_rows*5))
    axs = axs.flatten()
    # Visualize the mean error for each dimension in a bar chart
    for j in range(num_features):
        axs[j].bar(range(z_dim), feature_errors[j], tick_label=[f'{i+1}' for i in range(z_dim)])
        axs[j].set_title(f'{feat_labels[selected_features[j]]}', fontsize=20)
        axs[j].set_xlabel('Latent Dimension')
        axs[j].set_ylabel('Mean Error')

    # Remove extra axes
    for i in range(num_features, len(axs)):
        axs[i].remove()

    if savefile:
        plt.savefig(savefile)

    plt.close()
    return fig, selected_features


def visualize_latent_interpolation_chaos(vae, data, z_dim, feat_labels, num_features=3, savefile=None):
    """ Visualize the influence of each dimension in the latent space on individual features

    This function encodes the input data using the vae model, then for each dimension in the latent space it alters the
    latent vector by increasing the value of that dimension by 50% for each datapoint.
    The altered latent vector is then decoded and the error between the original and altered input is calculated.
    The mean error for each dimension is then plotted for each of the features being analyzed.

    The purpose of this visualization is to see which dimensions in the latent space have the most influence on the
    output of specific features.

    Args:
        vae: A trained VAE model
        data: A single batch of data to be used for the analysis
        z_dim: The number of dimensions in the latent space
        feat_labels: A list of the feature labels for the features being analyzed
        num_features: The number of features to analyze
        savefile (optional): The path to the file to save the visualization to

    Returns: The figure and a list of the selected features
    """
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
        z_altered[:, i] = 1.5 * z[:, i]

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
    # Visualize the mean error for each dimension in a bar chart
    for j in range(num_features):
        axs[j].bar(range(z_dim), feature_errors[j], tick_label=[f'{i+1}' for i in range(z_dim)])
        feat_name = feat_labels[selected_features[j]]
        # drop _thickness at end of feat_name
        if feat_name[-10:] == '_thickness':
            feat_name = feat_name[:-10]
        axs[j].set_title(f'{feat_name}', fontsize=25)
        axs[j].set_xlabel('Latent Dimension')
        axs[j].set_ylabel('Mean Error')

    # Remove extra axes
    for i in range(num_features, len(axs)):
        axs[i].remove()

    if savefile:
        plt.savefig(savefile)

    plt.close()
    return fig, selected_features


def visualize_errors_hist(vae, data, savefile=None):
    """ Creates a Histogram of the mean reconstruction errors.

    This function encodes the input data using the vae model, then decodes the latent space to generate a reconstruction
    of the input. The error between the original input and the reconstruction is then calculated. The mean error for
    each sample is then calculated and a histogram of the mean errors is plotted. A normal distribution is then fit to
    the histogram, and the mean and standard deviation of the distribution are printed. Using this information, the
    p-value of the mean error is calculated and printed. Using the p-value, the null hypothesis that the mean error is
    normally distributed is either accepted or rejected, which is represented by the color of the p-value line on the
    histogram.

    The purpose of this visualization is to see if the model has a particular bias in the reconstruction of the input.

    Args:
        vae: A trained VAE model
        data: A single batch of data to be used for the analysis
        savefile (optional): The path to the file to save the visualization to
    """

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
    title = "μ = %.2f,  σ = %.2f" % (mu, std)
    ax.set_title(title, fontsize=30)

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
    return fig


def visualize_feat_errors_hist(vae, data, savefile=None):
    """ Creates a Histogram of the mean reconstruction errors.

    This function encodes the input data using the vae model, then decodes the latent space to generate a reconstruction
    of the input. The error between the original input and the reconstruction is then calculated. The mean error for
    each sample is then calculated and a histogram of the mean errors is plotted. A normal distribution is then fit to
    the histogram, and the mean and standard deviation of the distribution are printed. Using this information, the
    p-value of the mean error is calculated and printed. Using the p-value, the null hypothesis that the mean error is
    normally distributed is either accepted or rejected, which is represented by the color of the p-value line on the
    histogram.

    The purpose of this visualization is to see if the model has a particular bias in the reconstruction of the input.

    Args:
        vae: A trained VAE model
        data: A single batch of data to be used for the analysis
        savefile (optional): The path to the file to save the visualization to
    """

    x, y = data
    x = tf.cast(x, dtype=tf.float32)
    # Generate the reconstruction
    z_mean, z_log_var, z, x_reconstruction = vae(data)

    # Calculate the error
    error = x - x_reconstruction

    # Calculate the mean error for each feature
    mean_errors = np.mean(error, axis=0)

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
    title = "μ = %.2f,  σ = %.2f" % (mu, std)
    ax.set_title(title, fontsize=30)

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
    return fig


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


def plot_training_results_hist(hist, save_path):
    plt.figure()
    plt.plot(hist.history['reconstruction_loss'], label='Training Loss')
    plt.plot(hist.history['val_reconstruction_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + '/recon_loss_hist.png')

    plt.figure()
    plt.plot(hist.history['kl_loss'], label='Training Loss')
    plt.plot(hist.history['val_kl_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + '/kl_loss_hist.png')

    plt.close()


def plot_training_results_cv(cv_results, save_path):
    plt.figure()
    plt.plot(cv_results['Training Reconstruction Loss History'], label='Training Loss')
    plt.plot(cv_results['Validation Reconstruction Loss History'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + '/recon_loss_cv.png')

    plt.figure()
    plt.plot(cv_results['Training KL Loss History'], label='Training Loss')
    plt.plot(cv_results['Validation KL Loss History'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + '/kl_loss_cv.png')

    plt.close()


def visualize_reconstruction_errors(vae, data, num_recon=6, random=False, idx=None, savefile=None):
    (x, y) = data
    z_mean, z_log_var, z, x_reconstruction = vae(data)

    if idx is None:
        if random:
            idx = np.random.choice(x.shape[0], num_recon, replace=False)
        else:
            idx = np.arange(num_recon)

    n_rows = int(np.ceil(num_recon/3))
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axs = axs.flatten()

    for i in range(num_recon):
        x_data = x[idx[i], :].numpy()
        x_recon = x_reconstruction[idx[i], :].numpy()

        # Fit linear regression model
        lr_model = LinearRegression()
        lr_model.fit(x_data.reshape(-1, 1), x_recon)

        # Predict y values
        y_pred = lr_model.predict(x_data.reshape(-1, 1))

        sns.scatterplot(x=x_data, y=x_recon, ax=axs[i], alpha=0.6)
        # Plot the diagonal (optimal fit)
        min_val = min(x_data.min(), x_recon.min())
        max_val = max(x_data.max(), x_recon.max())
        axs[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Optimal Fit')

        # Plot the linear regression fit
        axs[i].plot(x_data, y_pred, color='green', linestyle='--', label='Actual Fit')

        axs[i].set_title(f"Patient {idx[i]}", fontsize=25)
        axs[i].set_xlabel("Original")
        axs[i].set_ylabel("Reconstruction")
        axs[i].legend(loc='upper left') # Add a legend

    for i in range(num_recon, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile)

    plt.close()
    return fig


def visualize_feature_errors(vae, data, feat_labels, num_recon=6, random=False, idx=None, savefile=None):
    (x, y) = data
    z_mean, z_log_var, z, x_reconstruction = vae(data)

    if idx is None:
        if random:
            idx = np.random.choice(x.shape[1], num_recon, replace=False)
        else:
            idx = np.arange(num_recon)

    n_rows = int(np.ceil(num_recon/3))
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axs = axs.flatten()

    for i in range(num_recon):
        x_data = x[:, idx[i]].numpy()
        x_recon = x_reconstruction[:, idx[i]].numpy()

        # Fit linear regression model
        lr_model = LinearRegression()
        lr_model.fit(x_data.reshape(-1, 1), x_recon)

        # Predict y values
        y_pred = lr_model.predict(x_data.reshape(-1, 1))

        sns.scatterplot(x=x_data, y=x_recon, ax=axs[i], alpha=0.6)
        # Plot the diagonal (optimal fit)
        min_val = min(x_data.min(), x_recon.min())
        max_val = max(x_data.max(), x_recon.max())
        axs[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Optimal Fit')

        # Plot the linear regression fit
        axs[i].plot(x_data, y_pred, color='green', linestyle=':', label='Actual Fit')

        axs[i].set_title(f"{feat_labels[idx[i]]}", fontsize=25)
        axs[i].set_xlabel("Original")
        axs[i].set_ylabel("Reconstruction")
        axs[i].legend(loc='upper left') # Add a legend

    for i in range(num_recon, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile)

    plt.close()
    return fig


def top_feat_error_visualization(vae, data, feat_labels, savefile=None):
    x, y = data
    _, _, _, x_reconstruction = vae(data)

    # Calculate the mean squared error for each feature
    errors = [mean_squared_error(x[:, i].numpy(), x_reconstruction[:, i].numpy()) for i in range(x.shape[1])]
    idx = np.argsort(errors)[-6:]

    visualize_feature_errors(vae, data, feat_labels, len(idx), idx=idx, savefile=savefile)
    return errors


def top_recon_error_visualization(vae, data, savefile=None):
    x, y = data
    _, _, _, x_reconstruction = vae(data)

    # Calculate the mean squared error for each patient
    errors = [mean_squared_error(x[i].numpy(), x_reconstruction[i].numpy()) for i in range(len(x))]
    idx = np.argsort(errors)[-6:]

    visualize_reconstruction_errors(vae, data, len(idx), idx=idx, savefile=savefile)
    return errors

#%%
