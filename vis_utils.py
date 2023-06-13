from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd


def visualize_latent_space(vae, val_data, labels=None, savefile=None):
    val_batch_size = val_data.cardinality().numpy()
    val_data = val_data.batch(val_batch_size)
    val_batch = next(iter(val_data))

    # 1. Encode input samples
    z_mean, z_log_var, z, _ = vae(val_batch)
    z = z.numpy()

    # 2. Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z)

    # 3. Create a scatter plot
    plt.figure(figsize=(8, 6))
    if labels is None:
        sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], alpha=0.6)
    else:
        sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=labels, alpha=0.6,
                        palette=sns.color_palette('hsv', len(set(labels))))
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Visualization of Latent Space")
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()


def visualize_latent_space_multiple(vae_models, val_data, labels, savefile=None):
    val_batch_size = val_data.cardinality().numpy()
    val_data = val_data.batch(val_batch_size)
    val_batch = next(iter(val_data))

    z_results = []
    for i in range(len(vae_models)):
        model = vae_models[i]
        z_mean, z_log_var, z, _ = model(val_batch)
        z = z.numpy()
        tsne = TSNE(n_components=2, random_state=42)
        z_2d = tsne.fit_transform(z)
        plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6, label=labels[i])

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Visualization of Latent Space")
    plt.legend()

    if savefile is not None:
        plt.savefig(savefile)
    plt.show()


def latent_clustering(vae, val_data, num_clusters, savefile=None):
    # Assuming X is your high dimensional data.
    val_batch_size = val_data.cardinality().numpy()
    val_data = val_data.batch(val_batch_size)
    val_batch = next(iter(val_data))

    # 1. Encode input samples
    z_mean, z_log_var, z, _ = vae(val_batch)
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
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Component 1', y='Component 2', hue='Cluster', palette=sns.color_palette('hsv', num_clusters))

    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
    return labels


def plot_latent_dimensions(vae, val_data, z_dim, savefile=None):
    val_batch_size = val_data.cardinality().numpy()
    val_data = val_data.batch(val_batch_size)
    val_batch = next(iter(val_data))

    n_cols = 3
    n_rows = int(np.ceil(z_dim / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axs = axs.flatten()

    z_mean, z_log_var, z, _ = vae(val_batch)
    z = z.numpy()

    for i in range(z_dim):
        sns.histplot(z[:, i], ax=axs[i], kde=True, color='blue', alpha=0.6, bins=20)
        axs[i].set_title(f"Latent dimension {i+1}")

    for i in range(z_dim, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()


def plot_latent_dimensions_multiple(vae_models, val_data, z_dim, labels, savefile=None):
    val_batch_size = val_data.cardinality().numpy()
    val_data = val_data.batch(val_batch_size)
    val_batch = next(iter(val_data))

    n_cols = 3
    n_rows = int(np.ceil(z_dim / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    axs = axs.flatten()

    # Store the latent vectors for each model
    zs = []
    for vae in vae_models:
        z_mean, z_log_var, z, _ = vae(val_batch)
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
    plt.show()


def visualize_top_clusters(vae, val_data, top_cluster_indices, labels=None, savefile=None):
    val_batch_size = val_data.cardinality().numpy()
    val_data = val_data.batch(val_batch_size)
    val_batch = next(iter(val_data))

    # 1. Encode input samples
    z_mean, z_log_var, z, _ = vae(val_batch)
    z = z.numpy()

    # 2. Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z)

    # 3. Create color labels: '0' for non-top clusters and unique values for each top cluster
    color_labels = ['0' if i not in sum(top_cluster_indices.values(), [])
                    else str(list(top_cluster_indices.keys())[list(top_cluster_indices.values()).index(i)+1])
                    for i in range(len(z))]

    # 4. Create a scatter plot
    plt.figure(figsize=(8, 6))
    if labels is None:
        sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], alpha=0.6)
    else:
        sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=color_labels, alpha=0.6,
                        palette=sns.color_palette('hsv', len(set(color_labels))))  # Color palette for different
        # clusters
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Visualization of Latent Space")
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()
