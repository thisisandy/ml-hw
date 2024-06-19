import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.neighbors import KNeighborsClassifier

from datasets import load_dataset


# Function to set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# Set seeds for reproducibility
set_seed(42)

# Initialize Seaborn style
sns.set(style="whitegrid")


# Data preparation function
def load_and_prepare_data():
    dataset = load_dataset("imdb")
    X = dataset["train"]["text"]
    y = dataset["train"]["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train = vectorizer.fit_transform(X_train[:1000])
    X_test = vectorizer.transform(X_test[:1000])
    y_train = y_train[:1000]
    y_test = y_test[:1000]
    return X_train, X_test, y_train, y_test, X, y


# Model evaluation class
class KNNModelEvaluator:
    def __init__(self, model, name, output_dir="./hw1/output/im"):
        self.model = model
        self.name = name
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def plot_learning_curve(self, X, y):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_errors = 1 - np.mean(train_scores, axis=1)
        test_errors = 1 - np.mean(test_scores, axis=1)

        smoothed_train_errors = self.smooth(train_errors)
        smoothed_test_errors = self.smooth(test_errors)

        ax.set_title(f"Learning Curve: {self.name}", fontsize=18, weight="bold")
        ax.set_xlabel("Training examples", fontsize=14)
        ax.set_ylabel("Error rate", fontsize=14)
        ax.grid(True)
        ax.plot(
            train_sizes,
            smoothed_train_errors,
            color="r",
            label="Training error",
            linewidth=2,
        )
        ax.plot(
            train_sizes,
            smoothed_test_errors,
            color="g",
            label="Cross-validation error",
            linewidth=2,
        )
        ax.legend(loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        sns.despine()
        output_path = os.path.join(self.output_dir, "knn_learning_curve.png")
        plt.savefig(output_path)
        plt.close()

    def plot_model_complexity(self, X, y, param_name, param_range, plot_name_suffix):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        train_scores, test_scores = validation_curve(
            self.model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=10,
            scoring="accuracy",
            n_jobs=-1,
        )
        train_errors = 1 - np.mean(train_scores, axis=1)
        test_errors = 1 - np.mean(test_scores, axis=1)

        smoothed_train_errors = self.smooth(train_errors)
        smoothed_test_errors = self.smooth(test_errors)

        ax.plot(
            param_range,
            smoothed_train_errors,
            label="Training error",
            color="darkorange",
            lw=2,
        )
        ax.plot(
            param_range,
            smoothed_test_errors,
            label="Cross-validation error",
            color="navy",
            lw=2,
        )
        ax.set_title(
            f"Model Complexity: {self.name} ({plot_name_suffix})",
            fontsize=18,
            weight="bold",
        )
        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel("Error rate", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.grid(True)
        sns.despine()
        output_path = os.path.join(
            self.output_dir, f"knn_{param_name}_complexity_{plot_name_suffix}.png"
        )
        plt.savefig(output_path)
        plt.close()

    def smooth(self, values, smoothing_factor=0.9):
        smoothed_values = []
        last_value = values[0]
        for value in values:
            smoothed_value = last_value * smoothing_factor + value * (
                1 - smoothing_factor
            )
            smoothed_values.append(smoothed_value)
            last_value = smoothed_value
        return smoothed_values


# Main execution
def main():
    X_train, X_test, y_train, y_test, X, y = load_and_prepare_data()
    knn_model = KNeighborsClassifier(n_neighbors=10, weights="distance")
    model_evaluator = KNNModelEvaluator(knn_model, "k-Nearest Neighbors")
    model_evaluator.train(X_train, y_train)
    model_evaluator.plot_learning_curve(X_train, y_train)
    model_evaluator.plot_model_complexity(
        X_train, y_train, "n_neighbors", range(1, 21), "n_neighbors"
    )
    model_evaluator.plot_model_complexity(
        X_train, y_train, "leaf_size", range(10, 61, 10), "leaf_size"
    )


if __name__ == "__main__":
    main()
