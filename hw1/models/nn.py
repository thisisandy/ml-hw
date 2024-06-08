import os
import random

import matplotlib.pyplot as plt
import numpy as np
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


class EnhancedNN(pl.LightningModule):
    def __init__(self):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

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
        return optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def train_dataloader(self):
        # Load and preprocess data
        digits = datasets.load_digits()
        X = digits.images.reshape((len(digits.images), -1))
        y = digits.target
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
            y_train, dtype=torch.long
        )
        train_dataset = TensorDataset(X_train, y_train)
        return DataLoader(train_dataset, batch_size=64, shuffle=True)

    def val_dataloader(self):
        digits = datasets.load_digits()
        X = digits.images.reshape((len(digits.images), -1))
        y = digits.target
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(
            y_val, dtype=torch.long
        )
        val_dataset = TensorDataset(X_val, y_val)
        return DataLoader(val_dataset, batch_size=64, shuffle=False)


class ErrorRatePlotterCallback(pl.Callback):
    def __init__(self, smoothing_factor=0.1, output_dir="./output"):
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
        # Apply smoothing
        smoothed_train_error_rates = self.smooth(self.train_error_rates)
        smoothed_val_error_rates = self.smooth(self.val_error_rates)

        plt.figure(figsize=(10, 5))
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


# Create the model
model = EnhancedNN()

# Train the model
trainer = Trainer(max_epochs=100, callbacks=[ErrorRatePlotterCallback()])
trainer.fit(model)
