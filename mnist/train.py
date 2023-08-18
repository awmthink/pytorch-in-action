import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.utils import data
from torch.utils.data import random_split
from torchvision import datasets, transforms

from pytorch_lightning.cli import LightningCLI


class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_hiddens, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_classes)

    def forward(self, x):
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return self.fc2(F.relu(self.fc1(x)))


class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, num_classes),
        )

    def forward(self, x):
        return self.cnn(x)


class SimpleRNN(nn.Module):
    def __init__(
        self, rnn_type, input_size, num_hiddens, num_layers, bidirectional, num_classes
    ):
        super().__init__()
        self.hidden_size = num_hiddens
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        assert rnn_type in ("gru", "lstm"), "just support gru or lstm"
        self.num_directions = 2 if bidirectional else 1
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                bias=True,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.5,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=num_hiddens,
                num_layers=num_layers,
                bias=True,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.5,
            )
        self.fc = nn.Linear(num_hiddens * self.num_directions, num_classes)

    def forward(self, x):
        # (batch_size,1,28,28) -> (batch_size, 28, 28)
        x = torch.squeeze(x)
        h0 = torch.zeros(
            (self.num_layers * self.num_directions, x.shape[0], self.hidden_size),
            device=x.device,
        )
        if self.rnn_type == "gru":
            hidden_state = h0
        else:
            c0 = h0.clone()
            hidden_state = (h0, c0)
        # out: (batch_szie, sequence_len, hidden_size)
        out, _ = self.rnn(x, hidden_state)
        # out: (batch_szie, hidden_size)
        out = torch.mean(out, dim=1)
        return self.fc(out)


class LightningNN(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        # model
        self.model = model
        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict(
            {"train_loss": loss, "train_accuracy": accuracy},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict(
            {"val_loss": loss, "val_accuracy": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._common_step(batch, batch_idx)
        self.log_dict({"test_loss": loss, "test_accuracy": accuracy})
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)

        accuracy = self.accuracy(logits, y)
        return loss, accuracy

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self.forward(x)
        preds = logits.argmax(dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    cli = LightningCLI(LightningNN, MNISTDataModule)

# checkpoint_cb = ModelCheckpoint(
#     monitor="val_loss",
#     dirpath="s3://bucket101/checkpoints",
#     filename="simple-mnist-{epoch:02d}-{val_loss:.2f}",
#     save_top_k=3,
#     mode="min",
#     save_last=True,
# )

# early_stoppping_cb = EarlyStopping(monitor="val_loss")

# import torch

# torch.set_float32_matmul_precision("medium")

# data_dir = "../data/"
# accelerator = "gpu"
# devices = [0]
# input_size = 784
# num_classes = 10
# learning_rate = 0.001
# batch_size = 64
# num_epochs = 3
# num_threads = 4
# precision = "16-mixed"


# dm = MNISTDataModule(data_dir, batch_size, num_threads)
# model = LightningNN(input_size, num_classes, learning_rate)
# trainer = pl.Trainer(
#     default_root_dir="s3://bucket101",
#     accelerator=accelerator,
#     devices=devices,
#     min_epochs=1,
#     max_epochs=num_epochs,
#     precision=precision,
#     callbacks=[checkpoint_cb, early_stoppping_cb],
# )
# trainer.fit(model, dm)
