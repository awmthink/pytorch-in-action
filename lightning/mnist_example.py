from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.utils import data
from torch.utils.data import random_split
from torchvision import datasets, transforms

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback


class LightningNN(pl.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        # model
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def training_step(self, batch, batch_idx):
        loss, accuracy, f1_score = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy, f1_score = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": accuracy,
                "val_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, f1_score = self._common_step(batch, batch_idx)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_f1_score": f1_score,
            }
        )
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape((x.shape[0], -1))
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)

        accuracy = self.accuracy(logits, y)
        f1_score = self.f1_score(logits, y)

        return loss, accuracy, f1_score

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape((x.shape[0], -1))
        logits = self.forward(x)
        preds = logits.argmax(dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    # Download data on single GPU
    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        entire_dataset = datasets.MNIST(
            self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
        self.test_ds = datasets.MNIST(
            self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


checkpoint_cb = ModelCheckpoint(
    monitor="val_loss",
    dirpath="s3://bucket101/checkpoints",
    filename="simple-mnist-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
    save_last=True,
)

early_stoppping_cb = EarlyStopping(monitor="val_loss")

import torch

torch.set_float32_matmul_precision("medium")

data_dir = "../data/"
accelerator = "gpu"
devices = [0]
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3
num_threads = 4
precision = "16-mixed"


dm = MNISTDataModule(data_dir, batch_size, num_threads)
model = LightningNN(input_size, num_classes, learning_rate)
trainer = pl.Trainer(
    default_root_dir="s3://bucket101",
    accelerator=accelerator,
    devices=devices,
    min_epochs=1,
    max_epochs=num_epochs,
    precision=precision,
    callbacks=[checkpoint_cb, early_stoppping_cb],
)
trainer.fit(model, dm)
