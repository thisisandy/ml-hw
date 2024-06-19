import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

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


# Dataset and DataModule classes
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class IMDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        tokenizer,
        batch_size=8,
        max_length=512,
    ):
        super().__init__()
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        self.train_dataset = IMDBDataset(
            self.train_texts, self.train_labels, self.tokenizer, self.max_length
        )
        self.val_dataset = IMDBDataset(
            self.val_texts, self.val_labels, self.tokenizer, self.max_length
        )

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
    def __init__(self, n_classes=2, lr=2e-5):
        super(IMDBClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=n_classes
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log("train_error_rate", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        self.log("val_error_rate", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, correct_bias=False)
        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        return [optimizer], [scheduler]


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
        plt.title("Training and Validation Error Rate Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Error Rate")
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(
            self.output_dir, "imdb_transformer_training_validation_error_rate.png"
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

    train_texts = train_texts[:1000]
    train_labels = train_labels[:1000]
    val_texts = val_texts[:1000]
    val_labels = val_labels[:1000]

    return train_texts, train_labels, val_texts, val_labels


# Hyperparameter evaluation functions
def evaluate_hyperparameters(
    hyperparameter_values,
    hyperparameter_name,
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    tokenizer,
    output_dir="./hw1/output/im",
):
    results = []
    for value in hyperparameter_values:
        print(f"Evaluating {hyperparameter_name} = {value}")

        if hyperparameter_name == "batch_size":
            data_module = IMDBDataModule(
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                tokenizer,
                batch_size=value,
            )
            model = IMDBClassifier()
        elif hyperparameter_name == "learning_rate":
            data_module = IMDBDataModule(
                train_texts,
                train_labels,
                val_texts,
                val_labels,
                tokenizer,
                batch_size=8,
            )
            model = IMDBClassifier(lr=value)
        else:
            raise ValueError("Unsupported hyperparameter for evaluation")

        trainer = pl.Trainer(
            max_epochs=4,
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
    plt.title(f"Effect of {hyperparameter_name} on Error Rate")
    plt.xlabel(hyperparameter_name)
    plt.ylabel("Error Rate")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(
        output_dir, f"transformer_{hyperparameter_name}_tuning_results.png"
    )
    plt.savefig(output_path)
    plt.close()


# Main execution
if __name__ == "__main__":
    set_seed(42)

    train_texts, train_labels, val_texts, val_labels = load_and_prepare_data()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    batch_sizes = [4, 16, 64, 128]
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]

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
        tokenizer,
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
        tokenizer,
        output_dir,
    )
    plot_hyperparameter_tuning_results(
        learning_rate_results, "learning_rate", output_dir
    )
