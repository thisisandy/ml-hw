# %%

import os

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.random_projection import GaussianRandomProjection

sns.set_theme(style="whitegrid")
os.makedirs("./result", exist_ok=True)

# Load the Yeast dataset
yeast_data = fetch_openml(data_id=181, as_frame=True)
yeast_df = yeast_data.frame
yeast_df["target"] = yeast_data.target

# Identify categorical columns
categorical_columns = yeast_df.select_dtypes(include=["category"]).columns
non_categorical_columns = yeast_df.select_dtypes(exclude=["category"]).columns

# One-hot encode categorical columns
yeast_features = pd.get_dummies(yeast_df, columns=categorical_columns, drop_first=True)
# join the non-categorical columns
yeast_features = pd.concat([yeast_features, yeast_df[non_categorical_columns]], axis=1)

# Separate features and target
yeast_target = yeast_df["target"]

# Encode yeast target labels to numerical values
label_encoder = LabelEncoder()
yeast_target_encoded = label_encoder.fit_transform(yeast_target)

# Load the Breast Cancer dataset (keeping this for comparison)
bc_data = load_breast_cancer()
bc_df = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
bc_df["target"] = bc_data.target


def apply_clustering(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_data)

    em = GaussianMixture(n_components=n_clusters, random_state=42)
    em_labels = em.fit_predict(scaled_data)

    return kmeans_labels, em_labels


# Function to compute reconstruction error
def compute_reconstruction_error(original, reduced, inverse_transformer):
    reconstructed = inverse_transformer.inverse_transform(reduced)
    error = np.mean((original - reconstructed) ** 2)
    return error


# Function to map cluster labels to true labels
def map_clusters_to_labels(true_labels, cluster_labels):
    cost_matrix = np.zeros(
        (len(np.unique(cluster_labels)), len(np.unique(true_labels)))
    )
    for i in range(len(np.unique(cluster_labels))):
        for j in range(len(np.unique(true_labels))):
            cost_matrix[i, j] = np.sum((cluster_labels == i) & (true_labels == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    return col_ind[cluster_labels]


# Function to evaluate clustering
def evaluate_clustering(true_labels, kmeans_labels, em_labels):
    metrics = {}
    mapped_kmeans_labels = map_clusters_to_labels(true_labels, kmeans_labels)
    mapped_em_labels = map_clusters_to_labels(true_labels, em_labels)

    metrics["KMeans_ARI"] = adjusted_rand_score(true_labels, mapped_kmeans_labels)
    metrics["KMeans_NMI"] = normalized_mutual_info_score(
        true_labels, mapped_kmeans_labels
    )

    metrics["EM_ARI"] = adjusted_rand_score(true_labels, mapped_em_labels)
    metrics["EM_NMI"] = normalized_mutual_info_score(true_labels, mapped_em_labels)

    return metrics


# Apply clustering to Yeast dataset
yeast_kmeans_labels, yeast_em_labels = apply_clustering(yeast_features, n_clusters=10)
yeast_metrics = evaluate_clustering(
    yeast_target_encoded, yeast_kmeans_labels, yeast_em_labels
)

# Apply clustering to Breast Cancer dataset
bc_kmeans_labels, bc_em_labels = apply_clustering(
    bc_df[bc_data.feature_names], n_clusters=2
)
bc_metrics = evaluate_clustering(bc_df["target"], bc_kmeans_labels, bc_em_labels)

# Dimensionality Reduction
pca_yeast = PCA(n_components=15)
yeast_pca = pca_yeast.fit_transform(yeast_features)
pca_bc = PCA(n_components=10)
bc_pca = pca_bc.fit_transform(bc_df[bc_data.feature_names])

ica_yeast = FastICA(n_components=14, random_state=42)
yeast_ica = ica_yeast.fit_transform(yeast_features)
ica_bc = FastICA(n_components=28, random_state=42)
bc_ica = ica_bc.fit_transform(bc_df[bc_data.feature_names])

rp_yeast = GaussianRandomProjection(n_components=33, random_state=42)
yeast_rp = rp_yeast.fit_transform(yeast_features)
rp_bc = GaussianRandomProjection(n_components=29, random_state=42)
bc_rp = rp_bc.fit_transform(bc_df[bc_data.feature_names])

# Clustering on reduced data
yeast_pca_kmeans_labels, yeast_pca_em_labels = apply_clustering(
    yeast_pca, n_clusters=10
)
yeast_ica_kmeans_labels, yeast_ica_em_labels = apply_clustering(
    yeast_ica, n_clusters=10
)
yeast_rp_kmeans_labels, yeast_rp_em_labels = apply_clustering(yeast_rp, n_clusters=10)

bc_pca_kmeans_labels, bc_pca_em_labels = apply_clustering(bc_pca, n_clusters=2)
bc_ica_kmeans_labels, bc_ica_em_labels = apply_clustering(bc_ica, n_clusters=2)
bc_rp_kmeans_labels, bc_rp_em_labels = apply_clustering(bc_rp, n_clusters=2)

# Evaluate clustering on reduced data
yeast_pca_metrics = evaluate_clustering(
    yeast_target_encoded, yeast_pca_kmeans_labels, yeast_pca_em_labels
)
yeast_ica_metrics = evaluate_clustering(
    yeast_target_encoded, yeast_ica_kmeans_labels, yeast_ica_em_labels
)
yeast_rp_metrics = evaluate_clustering(
    yeast_target_encoded, yeast_rp_kmeans_labels, yeast_rp_em_labels
)

bc_pca_metrics = evaluate_clustering(
    bc_df["target"], bc_pca_kmeans_labels, bc_pca_em_labels
)
bc_ica_metrics = evaluate_clustering(
    bc_df["target"], bc_ica_kmeans_labels, bc_ica_em_labels
)
bc_rp_metrics = evaluate_clustering(
    bc_df["target"], bc_rp_kmeans_labels, bc_rp_em_labels
)

# Tabulate results
results = {
    "Dataset": [],
    "Method": [],
    "KMeans_ARI": [],
    "KMeans_NMI": [],
    "EM_ARI": [],
    "EM_NMI": [],
}

for dataset, metrics, method in [
    ("Yeast", yeast_metrics, "Original"),
    ("Yeast", yeast_pca_metrics, "PCA"),
    ("Yeast", yeast_ica_metrics, "ICA"),
    ("Yeast", yeast_rp_metrics, "RP"),
    ("Breast Cancer", bc_metrics, "Original"),
    ("Breast Cancer", bc_pca_metrics, "PCA"),
    ("Breast Cancer", bc_ica_metrics, "ICA"),
    ("Breast Cancer", bc_rp_metrics, "RP"),
]:
    results["Dataset"].append(dataset)
    results["Method"].append(method)
    results["KMeans_ARI"].append(metrics["KMeans_ARI"])
    results["KMeans_NMI"].append(metrics["KMeans_NMI"])
    results["EM_ARI"].append(metrics["EM_ARI"])
    results["EM_NMI"].append(metrics["EM_NMI"])

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_excel("./result/clustering_performance_comparison.xlsx", index=False)

print("Results saved to './result/clustering_performance_comparison.xlsx'")
