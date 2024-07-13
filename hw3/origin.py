# %%

import os

import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml, load_breast_cancer

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

# Load the Breast Cancer dataset (keeping this for comparison)
bc_data = load_breast_cancer()
bc_df = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
bc_df["target"] = bc_data.target


# plot the t-SNE visualization for yeast and bc on 1 * 2 grid

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
yeast_tsne_result = tsne.fit_transform(yeast_features)
bc_tsne_result = tsne.fit_transform(bc_df)

fig, axs = plt.subplots(1, 2, figsize=(20, 8))

sns.scatterplot(
    x=yeast_tsne_result[:, 0],
    y=yeast_tsne_result[:, 1],
    hue=yeast_target,
    palette="viridis",
    ax=axs[0],
    s=60,
    alpha=0.7,
    legend="brief",
)

sns.scatterplot(
    x=bc_tsne_result[:, 0],
    y=bc_tsne_result[:, 1],
    hue=bc_df["target"],
    palette="viridis",
    ax=axs[1],
    s=60,
    alpha=0.7,
    legend="brief",
)

# set legend font size
for ax in axs:
    ax.legend(fontsize=16)
axs[0].set_title("Yeast Dataset t-SNE Visualization", fontsize=16)
axs[1].set_title("Breast Cancer Dataset t-SNE Visualization", fontsize=16)
# compact layout
plt.tight_layout()

plt.savefig("./result/origin_tsne_visualization.png")
plt.show()
