import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ModelEvaluator:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"{self.name} accuracy: {accuracy:.2f}")
        print(f"Classification Report for {self.name}:\n{report}\n")
        return accuracy, y_pred

    def predict(self, X):
        return self.model.predict(X)

    def plot_learning_curve(self, X, y, ax):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        ax.set_title(self.name)
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.grid()
        ax.plot(
            train_sizes,
            np.mean(train_scores, axis=1),
            "o-",
            color="r",
            label="Training score",
        )
        ax.plot(
            train_sizes,
            np.mean(test_scores, axis=1),
            "o-",
            color="g",
            label="Cross-validation score",
        )
        ax.legend(loc="best")


def load_and_prepare_data():
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X, y


def plot_sample_predictions(X_test, y_test, models):
    # Select 10 random images from the test set
    np.random.seed(1)  # For reproducibility
    indices = np.random.choice(len(X_test), 10, replace=False)
    selected_images = X_test[indices]
    selected_labels = y_test[indices]

    fig, axes = plt.subplots(
        10, 4, figsize=(10, 20)
    )  # 10 images, 4 columns (Image, NN, kNN, SVM)
    fig.suptitle("Model Predictions on Test Images")

    for i, (image, label) in enumerate(zip(selected_images, selected_labels)):
        axes[i, 0].imshow(image.reshape(8, 8), cmap="gray")
        axes[i, 0].set_title(f"True: {label}")
        axes[i, 0].axis("off")

        for j, model_evaluator in enumerate(models, start=1):
            prediction = model_evaluator.predict(image.reshape(1, -1))
            axes[i, j].set_title(f"{model_evaluator.name}: {prediction[0]}")
            axes[i, j].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    X_train, X_test, y_train, y_test, X, y = load_and_prepare_data()

    # Define models
    models = [
        ModelEvaluator(
            MLPClassifier(
                hidden_layer_sizes=(20, 10),
                max_iter=1000,
                alpha=1e-4,
                solver="sgd",
                random_state=1,
                learning_rate_init=0.01,
            ),
            "Neural Network",
        ),
        ModelEvaluator(
            KNeighborsClassifier(n_neighbors=50, weights="distance"),
            "k-Nearest Neighbors",
        ),
        ModelEvaluator(SVC(kernel="rbf", C=100, gamma=0.001), "SVM"),
    ]

    # Train and evaluate models
    for model_evaluator in models:
        model_evaluator.train(X_train, y_train)
        model_evaluator.evaluate(X_test, y_test)

    # Plot learning curves
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle("Learning Curves for Different Classifiers")
    for i, model_evaluator in enumerate(models):
        model_evaluator.plot_learning_curve(X, y, axes[i])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Plot predictions on sample images
    plot_sample_predictions(X_test, y_test, models)


if __name__ == "__main__":
    main()
