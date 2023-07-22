import vis_utils as vu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import numpy as np
import tensorflow as tf


class VAEModelAnalyzer:
    def __init__(self, model, data, z, feat_labels, model_info=None):
        self.model = model
        self.data = data
        self.z = z
        self.feat_labels = feat_labels
        self.model_info = model_info
        self.model_results = {}

    def full_stack(self, save_path):
        # P1
        vu.visualize_latent_space(self.model, self.data, savefile=save_path + '/latent_space.png')
        vu.plot_latent_dimensions(self.model, self.data, z_dim=self.z, savefile=save_path + '/latent_dimensions.png')

        # P2
        num_clusters = 30
        k_mean = KMeans(num_clusters, n_init='auto', random_state=42)
        cluster_labels = k_mean.fit_predict(self.data[0])

        silhouette_vals = silhouette_samples(self.data[0], cluster_labels)

        silhouette_scores = []
        for i in range(num_clusters):
            score = np.mean(silhouette_vals[cluster_labels == i])
            silhouette_scores.append((i, score))

        top_clusters = sorted(silhouette_scores, key=lambda x: x[1], reverse=True)[:5]
        top_cluster_indices = []

        for idx, _ in top_clusters:
            top_indexes = np.where(cluster_labels == idx)[0]
            top_cluster_indices.append(top_indexes)

        vu.visualize_top_clusters(self.model, self.data, top_cluster_indices,
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




