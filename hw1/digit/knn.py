import os
import random

import matplotlib.pyplot as plt
import numpy as np
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


# Data preparation function
def load_and_prepare_data():
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val, X, y


# Class for evaluating and plotting boosting model performance
class BoostingModelEvaluator:
    def __init__(self, model, name, output_dir="./output/digit"):
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

        ax.set_title(f"Learning Curve: {self.name}", fontsize=16)
        ax.set_xlabel("Training examples", fontsize=14)
        ax.set_ylabel("Error rate", fontsize=14)
        ax.grid(True)
        ax.plot(
            train_sizes, train_errors, color="r", label="Training error", linewidth=2
        )
        ax.plot(
            train_sizes,
            test_errors,
            color="g",
            label="Cross-validation error",
            linewidth=2,
        )
        ax.legend(loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        output_path = os.path.join(self.output_dir, "boosting_learning_curve.png")
        plt.savefig(output_path)
        plt.close()

    # Function to evaluate boosting with decision trees
    def evaluate_boosting(X_train, y_train, X_val, y_val, n_estimators=50, max_depth=1):
        # Initialize base estimator
        base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        # Initialize AdaBoost
        ada_boost = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            algorithm="SAMME",
            random_state=42,
        )

        # Train AdaBoost
        ada_boost.fit(X_train, y_train)

        # Predict on validation data
        val_predictions = ada_boost.predict(X_val)

        # Calculate accuracy
        accuracy = accuracy_score(y_val, val_predictions)
        error_rate = 1 - accuracy

        return error_rate, ada_boost

    def plot_model_complexity(self, X, y, param_name, param_range):
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

        ax.plot(
            param_range, train_errors, label="Training error", color="darkorange", lw=2
        )
        ax.plot(
            param_range, test_errors, label="Cross-validation error", color="navy", lw=2
        )
        ax.set_title(f"Model Complexity: {self.name}", fontsize=16)
        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel("Error rate", fontsize=14)
        ax.legend(loc="best", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.grid(True)
        output_path = os.path.join(self.output_dir, "boosting_model_complexity.png")
        plt.savefig(output_path)
        plt.close()


# Main execution
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X, y = load_and_prepare_data()

    ada_boost_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50,
        algorithm="SAMME",
        random_state=42,
    )
    model_evaluator = BoostingModelEvaluator(
        ada_boost_model, "AdaBoost with Decision Trees"
    )
    model_evaluator.plot_learning_curve(X, y)
    model_evaluator.plot_model_complexity(X, y, "n_estimators", range(10, 101, 10))

    # Example with 50 estimators and max_depth=1
    _, ada_boost = evaluate_boosting(
        X_train, y_train, X_val, y_val, n_estimators=50, max_depth=1
    )

    # Plot feature importance
    feature_importances = ada_boost.feature_importances_
    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.title("Feature Importance from AdaBoost with Decision Trees")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.xticks(range(len(feature_importances)), range(len(feature_importances)))
    plt.grid(True)
    output_path = os.path.join("./output/digit", "boosting_feature_importance.png")
    plt.savefig(output_path)
    plt.close()
