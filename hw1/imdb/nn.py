import os
import random

import matplotlib.pyplot as plt
import numpy as np
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


# Set seeds for reproducibility
set_seed(42)


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
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


class IMDBClassifier(pl.LightningModule):
    def __init__(self, n_classes=2):
        super(IMDBClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=n_classes
        )
        self.criterion = torch.nn.CrossEntropyLoss()

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
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        error_rate = (preds != labels).float().mean()
        self.log("train_error_rate", error_rate, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        error_rate = (preds != labels).float().mean()
        self.log("val_error_rate", error_rate, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * trainer.max_epochs,
        )
        return [optimizer], [scheduler]


class ErrorRatePlotterCallback(pl.Callback):
    def __init__(self, smoothing_factor=0.1, output_dir="./imdb/output"):
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


def load_and_prepare_data():
    # Load the IMDB dataset using the datasets library
    dataset = load_dataset("imdb")
    train_texts, train_labels = dataset["train"]["text"], dataset["train"]["label"]
    val_texts, val_labels = dataset["test"]["text"], dataset["test"]["label"]
    return train_texts, train_labels, val_texts, val_labels


train_texts, train_labels, val_texts, val_labels = load_and_prepare_data()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create DataModule
data_module = IMDBDataModule(
    train_texts, train_labels, val_texts, val_labels, tokenizer, batch_size=8
)

# Create the model
model = IMDBClassifier()

# Train the model
trainer = pl.Trainer(
    max_epochs=3,
    callbacks=[ErrorRatePlotterCallback()],
    gpus=1 if torch.cuda.is_available() else 0,
)
trainer.fit(model, data_module)
