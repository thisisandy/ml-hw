# %%
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# Load and standardize the dataset
bc_data = load_breast_cancer()
bc_X = bc_data.data
scaler = StandardScaler()
bc_X_standardized = scaler.fit_transform(bc_X)
# %%
# Load the Yeast dataset
yeast_data = fetch_openml(data_id=181, as_frame=True)
yeast_df = yeast_data.frame
yeast_df["target"] = yeast_data.target
# %%
# Identify categorical columns
categorical_columns = yeast_df.select_dtypes(include=["category"]).columns
non_categorical_columns = yeast_df.select_dtypes(exclude=["category"]).columns

# One-hot encode categorical columns
yeast_features = pd.get_dummies(yeast_df, columns=categorical_columns, drop_first=True)
# join the non-categorical columns
yeast_features = pd.concat([yeast_features, yeast_df[non_categorical_columns]], axis=1)

yeast_X_standardized = scaler.fit_transform(yeast_features)
# %%
# Print the shape of the datasets
print(f"Yeast Dataset Shape: {yeast_X_standardized.shape}")
print(f"Breast Cancer Dataset Shape: {bc_X_standardized.shape}")
# %%
# Compute and print the rank of the datasets
yeast_rank = np.linalg.matrix_rank(yeast_X_standardized)
bc_rank = np.linalg.matrix_rank(bc_X_standardized)
print(f"Rank of Yeast Dataset: {yeast_rank}")
print(f"Rank of Breast Cancer Dataset: {bc_rank}")


# %%
# Compute number of components to capture 95% variance for PCA
def pca_components_for_variance(data, variance_threshold=0.95):
    data = StandardScaler().fit_transform(data)
    pca = PCA().fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    return num_components


yeast_pca_components = pca_components_for_variance(yeast_X_standardized)
bc_pca_components = pca_components_for_variance(bc_X_standardized)

print(
    f"Number of PCA components to capture 95% variance for Yeast dataset: {yeast_pca_components}"
)
print(
    f"Number of PCA components to capture 95% variance for Breast Cancer dataset: {bc_pca_components}"
)


# %%
# Compute number of components that have lowest reconstruction error for ICA
def ica_components(data):
    data_scaled = StandardScaler().fit_transform(data)
    X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)
    ica = FastICA(random_state=42, max_iter=1000, tol=1e-2)
    errors = []
    for n_components in range(2, data.shape[1] + 1):
        ica.set_params(n_components=n_components)
        ica.fit_transform(X_train)
        transformed_data = ica.transform(X_test)
        reconstructed_data = ica.inverse_transform(transformed_data)
        error = mean_squared_error(X_test, reconstructed_data)
        errors.append(error)
    return np.argmin(errors) + 1


yeast_ica_components = ica_components(yeast_X_standardized)
bc_ica_components = ica_components(bc_X_standardized)

print(f"Number of ICA components for Yeast dataset: {yeast_ica_components}")
print(f"Number of ICA components for Breast Cancer dataset: {bc_ica_components}")
# %%


# Compute reconstruction error for PCA, ICA, and Random Projection
def compute_reconstruction_error(method, data, n_components):
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)
    if method == "PCA":
        model = PCA(n_components=n_components)
    elif method == "ICA":
        model = FastICA(
            n_components=n_components, random_state=42, max_iter=10000, tol=1e-4
        )

    else:
        raise ValueError("Invalid method")

    model.fit(X_train)
    transformed_data = model.transform(X_test)

    reconstructed_data = model.inverse_transform(transformed_data)
    error = mean_squared_error(X_test, reconstructed_data)
    return error


# %%

yeast_pca_reconstruction_error = compute_reconstruction_error(
    "PCA", yeast_X_standardized, yeast_pca_components
)
yeast_ica_reconstruction_error = compute_reconstruction_error(
    "ICA", yeast_X_standardized, yeast_ica_components
)

bc_pca_reconstruction_error = compute_reconstruction_error(
    "PCA", bc_X_standardized, bc_pca_components
)
bc_ica_reconstruction_error = compute_reconstruction_error(
    "ICA", bc_X_standardized, bc_ica_components
)


# Print reconstruction errors
print(
    f"Reconstruction Error for Yeast dataset with PCA: {yeast_pca_reconstruction_error}"
)
print(
    f"Reconstruction Error for Yeast dataset with ICA: {yeast_ica_reconstruction_error}"
)


print(
    f"Reconstruction Error for Breast Cancer dataset with PCA: {bc_pca_reconstruction_error}"
)
print(
    f"Reconstruction Error for Breast Cancer dataset with ICA: {bc_ica_reconstruction_error}"
)
