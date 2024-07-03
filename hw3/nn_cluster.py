# %%
# nn_step5.py
# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import neural_network
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")
os.makedirs("./result", exist_ok=True)

# %%
# Load the Breast Cancer dataset
bc_data = load_breast_cancer()
X = bc_data.data
y = bc_data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode the target
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# Apply clustering algorithms
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels_train = kmeans.fit_predict(X_train)
kmeans_labels_test = kmeans.predict(X_test)

em = GaussianMixture(n_components=2, random_state=42)
em_labels_train = em.fit_predict(X_train)
em_labels_test = em.predict(X_test)

# Add cluster labels as new features
X_train_kmeans = np.column_stack((X_train, kmeans_labels_train))
X_test_kmeans = np.column_stack((X_test, kmeans_labels_test))
X_train_em = np.column_stack((X_train, em_labels_train))
X_test_em = np.column_stack((X_test, em_labels_test))
X_train_combined = np.column_stack((X_train, kmeans_labels_train, em_labels_train))
X_test_combined = np.column_stack((X_test, kmeans_labels_test, em_labels_test))

# %%
# Train a neural network classifier
NeuralNetwork = neural_network.MLPClassifier(max_iter=3000, random_state=42)
param_grid = {
    "hidden_layer_sizes": np.arange(2, 50, 2),
    "activation": ["logistic", "relu", "tanh"],
    "solver": ["adam", "sgd"],
    "alpha": [0.001, 0.01, 0.1],
    "learning_rate": ["invscaling", "adaptive"],
    "learning_rate_init": [0.001, 0.01, 0.1],
}


def fit_and_evaluate(X_train, X_test, y_train, y_test, method_name):
    NeuralNetworkCV = GridSearchCV(
        NeuralNetwork, param_grid, cv=5, n_jobs=-1, verbose=1
    )
    NeuralNetworkCV.fit(X_train, y_train)
    print(f"Best parameters ({method_name}): {NeuralNetworkCV.best_params_}")

    train_score = NeuralNetworkCV.score(X_train, y_train)
    test_score = NeuralNetworkCV.score(X_test, y_test)
    print(f"Train accuracy ({method_name}): {train_score:.4f}")
    print(f"Test accuracy ({method_name}): {test_score:.4f}")

    BestNeuralNetwork = NeuralNetworkCV.best_estimator_
    train_sizes, train_scores, test_scores = learning_curve(
        BestNeuralNetwork,
        X_train,
        y_train,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 20),
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(12, 8))
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="b",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="b", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="r", label="Cross-validation score"
    )
    plt.title(f"Learning Curve ({method_name})")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.savefig(f"./result/learning_curve_{method_name.lower()}.png")
    plt.show()

    return NeuralNetworkCV.best_estimator_, train_score, test_score


# %%
# Evaluate on original data with cluster labels
best_nn_kmeans, train_score_kmeans, test_score_kmeans = fit_and_evaluate(
    X_train_kmeans, X_test_kmeans, y_train, y_test, "KMeans Clusters"
)
# %%
best_nn_em, train_score_em, test_score_em = fit_and_evaluate(
    X_train_em, X_test_em, y_train, y_test, "EM Clusters"
)
# %%
best_nn_combined, train_score_combined, test_score_combined = fit_and_evaluate(
    X_train_combined, X_test_combined, y_train, y_test, "Combined Clusters"
)

# %%
# Compile results into a DataFrame
results = {
    "Method": ["Original", "KMeans Clusters", "EM Clusters", "Combined Clusters"],
    "Train Accuracy": [
        train_score_original,
        train_score_kmeans,
        train_score_em,
        train_score_combined,
    ],
    "Test Accuracy": [
        test_score_original,
        test_score_kmeans,
        test_score_em,
        test_score_combined,
    ],
}

results_df = pd.DataFrame(results)
print(results_df)

results_df.to_excel(
    "./result/nn_performance_with_clusters_comparison.xlsx", index=False
)

print("Results saved to './result/nn_performance_with_clusters_comparison.xlsx'")

# %%
