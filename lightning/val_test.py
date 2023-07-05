import os

import config
from model import ImageNetClassifier
from dataset import ImageNetDataModule

import torch
from torchvision import models

import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    transform = models.get_weight(config.weights).transforms()

    data_module = ImageNetDataModule(
        config.dataset_root_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        transform=transform,
    )

    ckpt = os.path.join(config.checkpoint_path, "epoch=25-step=4784.ckpt")
    model = ImageNetClassifier.load_from_checkpoint(ckpt)
    trainer = pl.Trainer(devices=1)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)
