"""
data.py

This module provides functions to load and prepare the IMDB dataset
for training and evaluating a machine learning model. The dataset is
preprocessed using TF-IDF vectorization to convert text data into numerical
features suitable for machine learning algorithms. It also includes
visualization functions for data distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from datasets import load_dataset


def load_and_prepare_data(num_samples=1000, test_size=0.25, random_state=0):
    """
    Load and prepare the IMDB dataset for training and testing.

    Parameters:
    - num_samples: int, default=1000
        The number of samples to use for training and testing.
    - test_size: float, default=0.25
        The proportion of the dataset to include in the test split.
    - random_state: int, default=0
        The seed used by the random number generator.

    Returns:
    - X_train: array-like, shape (num_samples * (1 - test_size), num_features)
        Training data features.
    - X_test: array-like, shape (num_samples * test_size, num_features)
        Testing data features.
    - y_train: array-like, shape (num_samples * (1 - test_size),)
        Training data labels.
    - y_test: array-like, shape (num_samples * test_size,)
        Testing data labels.
    - vectorizer: TfidfVectorizer
        The fitted TF-IDF vectorizer.
    """

    # Load the IMDB dataset
    dataset = load_dataset("imdb")

    # Extract the text and labels
    X = np.array(dataset["train"]["text"])
    y = np.array(dataset["train"]["label"])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Limit the number of samples for training and testing
    X_train = X_train[:num_samples]
    y_train = y_train[:num_samples]
    X_test = X_test[:num_samples]
    y_test = y_test[:num_samples]

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    return X_train, X_test, y_train, y_test, vectorizer


def visualize_data_distribution(X, y, vectorizer):
    """
    Visualize the distribution of the data.

    Parameters:
    - X: array-like, shape (num_samples, num_features)
        The features of the dataset.
    - y: array-like, shape (num_samples,)
        The labels of the dataset.
    - vectorizer: TfidfVectorizer
        The fitted TF-IDF vectorizer.
    """

    # Create a DataFrame for easier plotting
    text_lengths = np.sum(X > 0, axis=1)  # Count non-zero entries in the TF-IDF matrix
    data = {"text_length": text_lengths, "label": y}

    # Set up the plotting style
    sns.set(style="whitegrid")

    # Create a figure with subplots
    plt.figure(figsize=(14, 6))

    # Plot the distribution of text lengths
    plt.subplot(1, 2, 1)
    sns.histplot(data=data, x="text_length", bins=30, kde=True, color="blue")
    plt.title("Distribution of Text Lengths (Number of Words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    # Plot the distribution of class labels
    plt.subplot(1, 2, 2)
    sns.countplot(x="label", data=data, palette="viridis")
    plt.title("Distribution of Class Labels")
    plt.xlabel("Class Labels")
    plt.ylabel("Frequency")
    plt.xticks(ticks=[0, 1], labels=["Negative", "Positive"])

    plt.tight_layout()
    plt.savefig("./output/imdb/data_distribution.png", dpi=300)


if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test, vectorizer = load_and_prepare_data()
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Feature names:", vectorizer.get_feature_names_out()[:10])

    # Visualize the distribution of training data
    visualize_data_distribution(X_train, y_train, vectorizer)
