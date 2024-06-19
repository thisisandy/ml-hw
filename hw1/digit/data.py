"""
data_digits.py

This module provides functions to load and prepare the Digits dataset
for training and evaluating a machine learning model. The dataset is
standardized using StandardScaler to scale the feature values. It also includes
visualization functions for data distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(test_size=0.2, random_state=42):
    """
    Load and prepare the Digits dataset for training and testing.

    Parameters:
    - test_size: float, default=0.2
        The proportion of the dataset to include in the validation split.
    - random_state: int, default=42
        The seed used by the random number generator.

    Returns:
    - X_train: array-like, shape (num_samples * (1 - test_size), num_features)
        Training data features.
    - y_train: array-like, shape (num_samples * (1 - test_size),)
        Training data labels.
    - X_val: array-like, shape (num_samples * test_size, num_features)
        Validation data features.
    - y_val: array-like, shape (num_samples * test_size,)
        Validation data labels.
    - X: array-like, shape (num_samples, num_features)
        Full data features.
    - y: array-like, shape (num_samples,)
        Full data labels.
    """

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val, X, y


def visualize_data_distribution(X, y):
    """
    Visualize the distribution of the data.

    Parameters:
    - X: array-like, shape (num_samples, num_features)
        The features of the dataset.
    - y: array-like, shape (num_samples,)
        The labels of the dataset.
    """

    # Create a DataFrame for easier plotting
    data = {"feature_sum": np.sum(X, axis=1), "label": y}

    # Set up the plotting style
    sns.set_theme(style="whitegrid")

    # Create a figure with subplots
    plt.figure(figsize=(14, 6))

    # Plot the distribution of feature sums
    plt.subplot(1, 2, 1)
    sns.histplot(data=data, x="feature_sum", bins=30, kde=True, color="blue")
    plt.title("Distribution of Feature Sums")
    plt.xlabel("Sum of Feature Values")
    plt.ylabel("Frequency")

    # Plot the distribution of class labels
    plt.subplot(1, 2, 2)
    sns.countplot(x="label", data=data, palette="viridis")
    plt.title("Distribution of Class Labels")
    plt.xlabel("Class Labels")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("./hw1/output/digit/data_distribution.png", dpi=300)


if __name__ == "__main__":
    # Example usage
    X_train, y_train, X_val, y_val, X, y = load_and_prepare_data()
    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)

    # Visualize the distribution of the full data
    visualize_data_distribution(X, y)
