from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_latent_space(vae, val_data, savefile=None):
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
    plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.6)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Visualization of Latent Space")
    if savefile is not None:
        plt.savefig(savefile)
    plt.show()


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
