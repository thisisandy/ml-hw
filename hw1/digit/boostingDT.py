import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Function to set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# Set seeds for reproducibility
set_seed(42)


class BoostingModelEvaluator:
    def __init__(self, model, name, output_dir="./hw1/output/digit"):
        self.model = model
        self.name = name
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def plot_learning_curve(self, X, y):
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        train_errors = 1 - np.mean(train_scores, axis=1)
        test_errors = 1 - np.mean(test_scores, axis=1)

        smoothed_train_errors = self.smooth(train_errors)
        smoothed_test_errors = self.smooth(test_errors)

        ax.plot(
            train_sizes,
            smoothed_train_errors,
            color="red",
            label="Training error",
            linewidth=2,
        )
        ax.plot(
            train_sizes,
            smoothed_test_errors,
            color="green",
            label="Cross-validation error",
            linewidth=2,
        )

        ax.set_title(f"Learning Curve: {self.name}", fontsize=16)
        ax.set_xlabel("Training examples", fontsize=14)
        ax.set_ylabel("Error rate", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        output_path = os.path.join(self.output_dir, "boosting_learning_curve.png")
        plt.savefig(output_path)
        plt.close()

    def plot_model_complexity(self, X, y, param_name, param_range, plot_name_suffix):
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
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

        ax.set_title(f"Model Complexity: {self.name} ({plot_name_suffix})", fontsize=16)
        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel("Error rate", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        output_path = os.path.join(
            self.output_dir, f"boosting_model_complexity_{plot_name_suffix}.png"
        )
        plt.savefig(output_path)
        plt.close()

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


def load_and_prepare_data():
    digits = load_digits()
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X, y


def main():
    X_train, X_test, y_train, y_test, X, y = load_and_prepare_data()
    base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
    ada_boost = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,
        algorithm="SAMME",
        random_state=42,
    )
    model_evaluator = BoostingModelEvaluator(ada_boost, "AdaBoost with Decision Trees")
    model_evaluator.train(X_train, y_train)
    model_evaluator.plot_learning_curve(X, y)
    model_evaluator.plot_model_complexity(
        X, y, "n_estimators", [10, 20, 30, 50, 100], "n_estimators"
    )
    model_evaluator.plot_model_complexity(
        X_train,
        y_train,
        "learning_rate",
        np.logspace(-3, -1, 5),
        "learning_rate",
    )


if __name__ == "__main__":
    main()
