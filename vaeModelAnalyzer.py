import vis_utils as vu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import tensorflow as tf


class VAEModelAnalyzer:
    """ Class for analyses of VAE models

    This class can be used to perform a full analysis of a VAE model. The full-stack analysis includes the following:
        1. Latent Space Visualization
        2. Latent Dimension Visualization
        3. Top 5 Validation Clusters in Latent Space Visualization
        4. Latent Influence on Reconstruction Features Visualization
        5. Mean Latent Influence Visualization
        6. Reconstruction Error Histogram Visualization
        7. R2 Metric
        8. Reconstruction Error by Feature Metric

    The full functionality of these visualizations is described in the documentation of the vis_utils module.

    Attributes:
        model: A trained VAE model
        data: A single batch of validation data to be used for the analysis
        z: The dimensionality of the latent space
        feat_labels: The labels of the features in the data
        model_info (optional): A dictionary containing information about the model, such as the hyperparameters used
        model_results: A dictionary containing the metric results of the analysis
    """
    def __init__(self, model, data, z, feat_labels, model_info=None):
        self.model = model
        self.data = data
        self.z = z
        self.feat_labels = feat_labels
        self.model_info = model_info
        self.model_results = {}

    def full_stack(self, save_path):
        """ Perform a full-stack analysis of the model, generating all visualizations and metrics

        Args:
            save_path: The path to save the visualizations to
        """

        # P1
        vu.visualize_latent_space(self.model, self.data, savefile=save_path + '/latent_space.png')
        vu.plot_latent_dimensions(self.model, self.data, z_dim=self.z,
                                         savefile=save_path + '/latent_dimensions.png')

        # P2
        vu.visualize_top_clusters(self.model, self.data, num_clusters=30, top_k=5,
                                         savefile=save_path + '/top_5_clusters.png')

        # P3
        vu.visualize_latent_interpolation(self.model, self.data, feat_labels=self.feat_labels,
                                                        z_dim=self.z, savefile=save_path + '/latent_interpolation.png')
        vu.visualize_latent_influence(self.model, self.data, z_dim=self.z,
                                             savefile=save_path + '/latent_influence.png')

        # P4
        rec_data = tf.data.Dataset.from_tensor_slices(self.data)
        batched_data = rec_data.batch(rec_data.cardinality().numpy())
        _, self.model_results["R2"], _, _ = self.model.evaluate(batched_data)
        vis6 = vu.visualize_errors_hist(self.model, self.data, savefile=save_path + '/errors_hist.png')
        self.model_results["Feature Errors"] = vu.calc_feature_errors(self.model, self.data,
                                                                      feat_labels=self.feat_labels,
                                                                      savefile=save_path + '/feature_errors.csv')

        # Full Stack Visualization
        fig, axs = plt.subplots(3, 2, figsize=(15, 20))
        axs = axs.flatten()
        axs[0].imshow(imread(save_path + '/latent_space.png'))
        axs[0].set_title('Latent Space')
        axs[0].axis('off')
        axs[1].imshow(imread(save_path + '/latent_dimensions.png'))
        axs[1].set_title('Latent Dimensions')
        axs[1].axis('off')
        axs[2].imshow(imread(save_path + '/top_5_clusters.png'))
        axs[2].set_title('Top 5 Validation Clusters in Latent Space')
        axs[2].axis('off')
        axs[3].imshow(imread(save_path + '/latent_interpolation.png'))
        axs[3].set_title('Latent Influence on Reconstruction Features')
        axs[3].axis('off')
        axs[4].imshow(imread(save_path + '/latent_influence.png'))
        axs[4].set_title('Mean Latent Influence')
        axs[4].axis('off')
        axs[5].imshow(imread(save_path + '/errors_hist.png'))
        axs[5].set_title('Reconstruction Error Histogram')
        axs[5].axis('off')
        plt.tight_layout()

        plt.savefig(save_path + '/full_stack.png')
        plt.close()




