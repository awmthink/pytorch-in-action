import os

import config
from model import ImageNetClassifier
from dataset import ImageNetDataModule, FlowerDataModule, ImageNetDataModule1
from callbacks import MyPrintingCallback

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    data_module = ImageNetDataModule(
        config.dataset_root_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    pretrained_weights = None
    if config.pretrain:
        pretrained_weights = config.weights
    model = ImageNetClassifier(config.model, pretrained_weights, config.learning_rate)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.checkpoint_path,
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    early_stoppping_cb = EarlyStopping(monitor="val_loss")

    # logger = TensorBoardLogger("tb_logs", name="imagenet_model_v1")
    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    # )

    trainer = pl.Trainer(
        logger=TensorBoardLogger(config.log_dir),
        profiler=config.profiler,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        min_epochs=1,
        max_epochs=config.num_epochs,
        precision=config.precision,
        callbacks=[checkpoint_cb, early_stoppping_cb, MyPrintingCallback()],
    )

    resume_ckpt = os.path.join(config.checkpoint_path, "last.ckpt")
    if not os.path.exists(resume_ckpt):
        resume_ckpt = None

    trainer.fit(model, data_module, ckpt_path=resume_ckpt)
