import vis_utils as vu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import numpy as np
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
        vu.plot_latent_dimensions(self.model, self.data, z_dim=self.z, savefile=save_path + '/latent_dimensions.png')

        # P2
        vu.visualize_top_clusters(self.model, self.data, num_clusters=30, top_k=5,
                                  savefile=save_path + '/top_5_clusters.png')

        # P3
        vu.visualize_latent_interpolation(self.model, self.data, feat_labels=self.feat_labels, z_dim=self.z,
                                          savefile=save_path + '/latent_interpolation.png')
        vu.visualize_latent_influence(self.model, self.data, z_dim=self.z, savefile=save_path + '/latent_influence.png')

        # P4
        rec_data = tf.data.Dataset.from_tensor_slices(self.data)
        batched_data = rec_data.batch(rec_data.cardinality().numpy())
        _, self.model_results["R2"], _, _ = self.model.evaluate(batched_data)
        vu.visualize_errors_hist(self.model, self.data, savefile=save_path + '/errors_hist.png')
        self.model_results["Feature Errors"] = vu.calc_feature_errors(self.model, self.data,
                                                                      feat_labels=self.feat_labels,
                                                                      savefile=save_path + '/feature_errors.csv')




