import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset


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

# Initialize Seaborn style
sns.set(style="whitegrid")


# Dataset and DataModule classes
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return self.texts.shape[0]

    def __getitem__(self, idx):
        text = self.texts[idx].toarray()[0]
        label = self.labels[idx]
        return {
            "text": torch.tensor(text, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, train_texts, train_labels, val_texts, val_labels, batch_size=8):
        super().__init__()
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = IMDBDataset(self.train_texts, self.train_labels)
        self.val_dataset = IMDBDataset(self.val_texts, self.val_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )


# Model and Callback classes
class IMDBClassifier(pl.LightningModule):
    def __init__(self, input_dim, n_classes=2, lr=0.001, dropout=0.5):
        super(IMDBClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )
        self.lr = lr
        self.dropout = dropout

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        texts = batch["text"]
        labels = batch["label"]
        outputs = self(texts)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        torch.argmax(outputs, dim=1)
        self.log("train_error_rate", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        texts = batch["text"]
        labels = batch["label"]
        outputs = self(texts)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("val_error_rate", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class ErrorRatePlotterCallback(pl.Callback):
    def __init__(self, smoothing_factor=0.1, output_dir="./hw1/output/im"):
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

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Error Rate", fontsize=14)
        plt.legend()
        plt.grid(True)
        sns.despine()
        dropout = pl_module.dropout if hasattr(pl_module, "dropout") else 0.5
        lr = pl_module.lr if hasattr(pl_module, "lr") else 0.01
        plt.title(
            f"Error Rate vs. Epoch (Dropout={dropout}, LR={lr}",
            fontsize=18,
            weight="bold",
        )
        output_path = os.path.join(
            self.output_dir, f"./nn/{dropout}_{lr}_error_rate.png"
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


# Data preparation function
def load_and_prepare_data():
    dataset = load_dataset("imdb")
    train_texts, train_labels = dataset["train"]["text"], dataset["train"]["label"]
    val_texts, val_labels = dataset["test"]["text"], dataset["test"]["label"]

    train_texts, train_labels = train_texts[:1000], train_labels[:1000]
    val_texts, val_labels = val_texts[:1000], val_labels[:1000]

    vectorizer = TfidfVectorizer(max_features=5000)
    train_texts = vectorizer.fit_transform(train_texts)
    val_texts = vectorizer.transform(val_texts)

    return train_texts, train_labels, val_texts, val_labels, train_texts.shape[1]


# Hyperparameter evaluation functions
def evaluate_hyperparameters(
    hyperparameter_values,
    hyperparameter_name,
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    input_dim,
    output_dir="./hw1/output/im",
):
    results = []
    for value in hyperparameter_values:
        print(f"Evaluating {hyperparameter_name} = {value}")

        if hyperparameter_name == "batch_size":
            data_module = IMDBDataModule(
                train_texts, train_labels, val_texts, val_labels, batch_size=value
            )
            model = IMDBClassifier(input_dim=input_dim)
        elif hyperparameter_name == "learning_rate":
            data_module = IMDBDataModule(
                train_texts, train_labels, val_texts, val_labels, batch_size=8
            )
            model = IMDBClassifier(input_dim=input_dim, lr=value)
        elif hyperparameter_name == "dropout":
            data_module = IMDBDataModule(
                train_texts, train_labels, val_texts, val_labels, batch_size=8
            )
            model = IMDBClassifier(input_dim=input_dim, dropout=value)
        else:
            raise ValueError("Unsupported hyperparameter for evaluation")

        trainer = pl.Trainer(
            max_epochs=6,
            callbacks=[ErrorRatePlotterCallback(output_dir=output_dir)],
            accelerator="gpu",
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

    return results


def plot_hyperparameter_tuning_results(
    results, hyperparameter_name, output_dir="./hw1/output/im"
):
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(
        df[hyperparameter_name], df["train_error_rate"], label="Training Error Rate"
    )
    plt.plot(
        df[hyperparameter_name], df["val_error_rate"], label="Validation Error Rate"
    )
    plt.title(
        f"Effect of {hyperparameter_name} on Error Rate", fontsize=18, weight="bold"
    )
    plt.xlabel(hyperparameter_name, fontsize=14)
    plt.ylabel("Error Rate", fontsize=14)
    plt.legend()
    plt.grid(True)
    sns.despine()

    output_path = os.path.join(
        output_dir, f"nn_{hyperparameter_name}_tuning_results.png"
    )
    plt.savefig(output_path)
    plt.close()


# Main execution
if __name__ == "__main__":
    set_seed(42)

    train_texts, train_labels, val_texts, val_labels, input_dim = (
        load_and_prepare_data()
    )

    batch_sizes = [4, 8, 16, 32]
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    dropout_rates = [0.2, 0.4, 0.5, 0.6, 0.8]

    output_dir = "./hw1/output/im"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    batch_size_results = evaluate_hyperparameters(
        batch_sizes,
        "batch_size",
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        input_dim,
        output_dir,
    )
    plot_hyperparameter_tuning_results(batch_size_results, "batch_size", output_dir)

    learning_rate_results = evaluate_hyperparameters(
        learning_rates,
        "learning_rate",
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        input_dim,
        output_dir,
    )
    plot_hyperparameter_tuning_results(
        learning_rate_results, "learning_rate", output_dir
    )

    dropout_rate_results = evaluate_hyperparameters(
        dropout_rates,
        "dropout",
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        input_dim,
        output_dir,
    )
    plot_hyperparameter_tuning_results(dropout_rate_results, "dropout", output_dir)
