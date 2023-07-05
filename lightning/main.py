from model import ImageNetClassifier
from dataset import ImageNetDataModule

import torch

from pytorch_lightning.cli import LightningCLI

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    cli = LightningCLI(
        ImageNetClassifier,
        ImageNetDataModule,
        parser_kwargs={
            "fit": {"default_config_files": ["config/fit.yaml"]},
            "test": {"default_config_files": ["config/test.yaml"]},
        },
    )
