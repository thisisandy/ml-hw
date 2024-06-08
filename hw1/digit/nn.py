import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# Function to set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


# Set seeds for reproducibility
set_seed(42)


# Data preparation function
def load_and_prepare_data():
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        y_train, dtype=torch.long
    )
    X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(
        y_val, dtype=torch.long
    )
    return X_train, X_val, y_train, y_val


class EnhancedNN(pl.LightningModule):
    def __init__(self, lr=0.01, dropout=0.5):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lr = lr

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        error_rate = (preds != y).float().mean()
        self.log("train_error_rate", error_rate, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        error_rate = (preds != y).float().mean()
        self.log("val_error_rate", error_rate, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)


# Custom data module
class DigitsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.X_train, self.X_val, self.y_train, self.y_val = load_and_prepare_data()

    def train_dataloader(self):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = TensorDataset(self.X_val, self.y_val)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)


# Callback to plot error rates
class ErrorRatePlotterCallback(pl.Callback):
    def __init__(self, smoothing_factor=0.1, output_dir="./digit/output"):
        super().__init__()
        self.train_error_rates = []
        self.val_error_rates = []
        self.smoothing_factor = smoothing_factor
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        train_error_rate = trainer.callback_metrics.get("train_error_rate")
        if train_error_rate is not None:
            self.train_error_rates.append(train_error_rate.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_error_rate = trainer.callback_metrics.get("val_error_rate")
        if val_error_rate is not None:
            self.val_error_rates.append(val_error_rate.item())

    def on_train_end(self, trainer, pl_module):
        smoothed_train_error_rates = self.smooth(self.train_error_rates)
        smoothed_val_error_rates = self.smooth(self.val_error_rates)

        plt.figure(figsize=(10, 5), dpi=300)
        plt.plot(smoothed_train_error_rates, label="Training Error Rate")
        plt.plot(smoothed_val_error_rates, label="Validation Error Rate")
        plt.title("Training and Validation Error Rate Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Error Rate")
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(
            self.output_dir, "nn_training_validation_error_rate.png"
        )
        plt.savefig(output_path)
        plt.close()

    def smooth(self, values):
        smoothed_values = []
        last_value = values[0]
        for value in values:
            smoothed_value = last_value * self.smoothing_factor + value * (
                1 - self.smoothing_factor
            )
            smoothed_values.append(smoothed_value)
            last_value = smoothed_value
        return smoothed_values


# Function to evaluate hyperparameters and plot complexity curves
def evaluate_hyperparameters(
    hyperparameter_values, hyperparameter_name, output_dir="./digit/output"
):
    results = []

    for value in hyperparameter_values:
        if hyperparameter_name == "lr":
            model = EnhancedNN(lr=value)
        elif hyperparameter_name == "dropout":
            model = EnhancedNN(dropout=value)
        else:
            model = EnhancedNN()

        if hyperparameter_name == "batch_size":
            data_module = DigitsDataModule(batch_size=value)
        else:
            data_module = DigitsDataModule()

        trainer = Trainer(
            max_epochs=100, callbacks=[ErrorRatePlotterCallback()], logger=False
        )
        trainer.fit(model, data_module)

        train_error_rate = trainer.callback_metrics.get("train_error_rate", None)
        val_error_rate = trainer.callback_metrics.get("val_error_rate", None)
        if train_error_rate is not None and val_error_rate is not None:
            results.append(
                {
                    hyperparameter_name: value,
                    "train_error_rate": train_error_rate.item(),
                    "val_error_rate": val_error_rate.item(),
                }
            )

    plot_hyperparameter_tuning_results(results, hyperparameter_name, output_dir)


# Function to plot hyperparameter tuning results
def plot_hyperparameter_tuning_results(
    results, hyperparameter_name, output_dir="./digit/output"
):
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(
        df[hyperparameter_name], df["train_error_rate"], label="Training Error Rate"
    )
    plt.plot(
        df[hyperparameter_name], df["val_error_rate"], label="Validation Error Rate"
    )
    plt.title(f"Effect of {hyperparameter_name} on Error Rate")
    plt.xlabel(hyperparameter_name)
    plt.ylabel("Error Rate")
    plt.legend()
    plt.grid(True)
    output_path = os.path.join(output_dir, f"{hyperparameter_name}_tuning_results.png")
    plt.savefig(output_path)
    plt.close()


# Main execution
def main():
    DigitsDataModule()

    lr_values = [0.001, 0.01, 0.1, 0.5, 1.0]
    evaluate_hyperparameters(lr_values, "lr")

    batch_sizes = [16, 32, 64, 128]
    evaluate_hyperparameters(batch_sizes, "batch_size")

    dropout_values = [0.2, 0.4, 0.5, 0.6, 0.8]
    evaluate_hyperparameters(dropout_values, "dropout")


if __name__ == "__main__":
    main()
