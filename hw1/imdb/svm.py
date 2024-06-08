import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.svm import SVC

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
    X_train_tfidf = vectorizer.fit_transform(X_train[:1000])
    X_test_tfidf = vectorizer.transform(X_test[:1000])
    y_train = y_train[:1000]
    y_test = y_test[:1000]
    return X_train_tfidf, X_test_tfidf, y_train, y_test


# Model evaluation class
class SVMModelEvaluator:
    def __init__(self, model, name, output_dir="./output/im"):
        self.model = model
        self.name = name
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def plot_learning_curve(self, X, y, ax=None):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_errors = 1 - np.mean(train_scores, axis=1)
        test_errors = 1 - np.mean(test_scores, axis=1)

        smoothed_train_errors = self.smooth(train_errors)
        smoothed_test_errors = self.smooth(test_errors)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(
            train_sizes,
            smoothed_train_errors,
            label=f"{self.name} - Training error",
            linewidth=2,
        )
        ax.plot(
            train_sizes,
            smoothed_test_errors,
            label=f"{self.name} - Cross-validation error",
            linewidth=2,
        )
        ax.set_xlabel("Training examples", fontsize=14)
        ax.set_ylabel("Error rate", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True)
        sns.despine()

    def plot_model_complexity(self, X, y, param_name, param_range, ax=None):
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

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(
            param_range,
            smoothed_train_errors,
            label=f"{self.name} - Training error",
            linewidth=2,
        )
        ax.plot(
            param_range,
            smoothed_test_errors,
            label=f"{self.name} - Cross-validation error",
            linewidth=2,
        )
        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel("Error rate", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True)
        sns.despine()

    def smooth(self, values, smoothing_factor=0.1):
        smoothed_values = []
        last_value = values[0]
        for value in values:
            smoothed_value = last_value * smoothing_factor + value * (
                1 - smoothing_factor
            )
            smoothed_values.append(smoothed_value)
            last_value = smoothed_value
        return smoothed_values


# Function to plot comparison of kernels
def plot_kernel_comparison(X_train, y_train, param_range, output_dir="./output/im"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=300)

    linear_svm = SVC(kernel="linear", C=1.0)
    rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale")

    linear_evaluator = SVMModelEvaluator(linear_svm, "Linear SVM", output_dir)
    rbf_evaluator = SVMModelEvaluator(rbf_svm, "RBF SVM", output_dir)

    linear_evaluator.plot_learning_curve(X_train, y_train, ax=axes[0])
    rbf_evaluator.plot_learning_curve(X_train, y_train, ax=axes[0])
    axes[0].set_title("Learning Curve Comparison", fontsize=16)

    linear_evaluator.plot_model_complexity(
        X_train, y_train, "C", param_range, ax=axes[1]
    )
    rbf_evaluator.plot_model_complexity(X_train, y_train, "C", param_range, ax=axes[1])
    axes[1].set_title("Model Complexity Comparison", fontsize=16)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "svm_kernel_comparison.png")
    plt.savefig(output_path)
    plt.close()


# Main execution
def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    param_range = np.linspace(0.001, 1, 20)

    plot_kernel_comparison(X_train, y_train, param_range)


if __name__ == "__main__":
    main()
