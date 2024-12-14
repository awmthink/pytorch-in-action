import os
import sys
import shutil
import logging
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR, LRScheduler
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, skip_first_batches
from accelerate.utils import set_seed
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
import simple_parsing
from timm import create_model

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    name: str = "resnet50"
    pretrained: bool = False


@dataclass
class DatasetArguments:
    data_root: str


@dataclass
class TrainingArguments:
    output_dir: str
    seed: Optional[int] = None
    fp16: bool = False
    bf16: bool = False
    max_grad_norm: float = 1.0
    learning_rate: float = 0.1
    train_batch_size_per_device: int = 8
    eval_batch_size_per_device: int = 8
    grad_accumulation_batches: int = 1
    weight_decay: float = 0
    momentum: float = 0.9
    num_train_epochs: int = 3
    logging_steps: int = 500
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_safetensors: bool = False
    save_total_limit: Optional[int] = 3
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    resume_from_checkpoint: Optional[str] = None


class ImageNetDataset(Dataset):
    """ImageNet dataset that loads images and labels from a data root directory."""

    def __init__(self, data_root: str, split: str = "train", transform=None):
        if split not in ["train", "val"]:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")

        self.split = split
        self.transform = transform
        self.image_root = os.path.join(data_root, split)
        self.img_lst, self.labels = self._load_meta(
            os.path.join(data_root, f"meta/{split}.txt")
        )

    def _load_meta(self, meta_file_path: str):
        """Load image paths and labels from meta file."""
        with open(meta_file_path, "r", encoding="utf-8") as f:
            data = [line.strip().split() for line in f]
        return zip(*[(img, int(label)) for img, label in data])

    def __getitem__(self, index):
        """Get image and label at index."""
        img_path = os.path.join(self.image_root, self.img_lst[index])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[index], dtype=torch.int64)

    def __len__(self):
        return len(self.img_lst)


def prepare_datalader(dataset_args, training_args):
    """Prepare train and validation dataloaders."""
    # 标准化变换，使用ImageNet数据集的均值和标准差
    # mean: [0.485, 0.456, 0.406] - ImageNet RGB通道的均值
    # std: [0.229, 0.224, 0.225] - ImageNet RGB通道的标准差
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 训练数据的数据增强变换
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),  # 随机裁剪并缩放到224x224
            T.RandomHorizontalFlip(),  # 随机水平翻转
            T.ToTensor(),  # 转换为tensor，并归一化到[0,1]
            normalize,  # 使用ImageNet统计量标准化
        ]
    )
    train_data = ImageNetDataset(dataset_args.data_root, "train", train_transform)

    # 验证数据的变换（不需要数据增强）
    val_transform = T.Compose(
        [
            T.Resize(256),  # 将短边缩放到256
            T.CenterCrop(224),  # 中心裁剪224x224
            T.ToTensor(),  # 转换为tensor，并归一化到[0,1]
            normalize,  # 使用ImageNet统计量标准化
        ]
    )
    val_data = ImageNetDataset(dataset_args.data_root, "val", val_transform)

    train_loader = DataLoader(
        train_data,
        training_args.train_batch_size_per_device,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_data,
        training_args.eval_batch_size_per_device,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, val_loader


def _compute_topk_matches(logits, labels, k):
    _, preds = torch.topk(logits, k, dim=-1)
    return (preds == labels.unsqueeze(-1)).sum()


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        training_args: TrainingArguments,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: LRScheduler,
    ):
        self.model = model
        self.training_args = training_args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.accelerator = Accelerator(
            project_dir=self.training_args.output_dir,
            log_with="tensorboard",
            gradient_accumulation_steps=training_args.grad_accumulation_batches,
            mixed_precision=(
                "fp16" if training_args.fp16 else "bf16" if training_args.bf16 else "no"
            ),
            step_scheduler_with_optimizer=False,
        )
        self.accelerator.init_trackers("tensorboard")

        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_loader,
            self.val_loader,
        )

        self.epoch = 0
        self.global_steps = 0

        self.total_steps = (
            self.training_args.num_train_epochs
            * len(self.train_loader)
            // self.training_args.grad_accumulation_batches
        )

    def train(self):
        self.maybe_load_checkpoint()

        batch_idx = self.global_steps * self.training_args.grad_accumulation_batches
        start_epoch = batch_idx // len(self.train_loader) + 1
        skip_batches = batch_idx % len(self.train_loader)

        self.training_bar = tqdm(
            total=self.total_steps,
            dynamic_ncols=True,
            unit="step",
            disable=not self.accelerator.is_main_process,
        )
        self.training_bar.set_description("Training")
        self.training_bar.update(self.global_steps)

        for epoch in range(start_epoch, self.training_args.num_train_epochs + 1):
            self.epoch = epoch

            if skip_batches > 0 and epoch == start_epoch:
                active_dataloader = skip_first_batches(self.train_loader, skip_batches)
            else:
                active_dataloader = self.train_loader

            for input_tensors, labels in active_dataloader:
                with self.accelerator.accumulate():
                    with self.accelerator.autocast():
                        logits = self.model(input_tensors)
                        loss = self.criterion(logits, labels)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_steps += 1

                    self.loging_training_metrics(loss)
                    self.evaluate_and_logging_metrics()
                    self.save_checkpoints()
                    self.training_bar.update()
            self.lr_scheduler.step()

    def loging_training_metrics(self, loss):
        if self.global_steps % self.training_args.logging_steps != 0:
            return
        if not self.accelerator.is_main_process:
            return
        self.training_bar.write(
            f"Train: epoch {self.epoch} / {self.training_args.num_train_epochs}, "
            f"steps: {self.global_steps} / {self.training_bar.total}, "
            f"training loss: {loss.item():.4f}"
        )
        self.accelerator.log({"training loss": loss.item()}, self.global_steps)

    @torch.inference_mode()
    def evaluate(self):
        device = self.accelerator.device
        top1_num_matches = torch.tensor(0, device=device)
        top5_num_matches = torch.tensor(0, device=device)
        total_loss = torch.tensor(0.0, device=device)

        prediction_bar = tqdm(
            desc="Evaluating",
            total=len(self.val_loader),
            leave=False,
            dynamic_ncols=True,
            unit="batch",
            disable=not self.accelerator.is_main_process,
        )

        for input_tensors, labels in self.val_loader:
            with self.accelerator.autocast():
                logits = self.model(input_tensors)
                loss = self.criterion(logits, labels)
            total_loss += loss
            top1_num_matches += _compute_topk_matches(logits, labels, k=1)
            top5_num_matches += _compute_topk_matches(logits, labels, k=5)

            prediction_bar.update()

        total_loss = self.accelerator.reduce(total_loss, "mean")
        top1_num_matches = self.accelerator.reduce(top1_num_matches)
        top5_num_matches = self.accelerator.reduce(top5_num_matches)

        return {
            "loss": total_loss.item() / len(self.val_loader),
            "accuracy@top1": top1_num_matches.item() / len(self.val_loader.dataset),
            "accuracy@top5": top5_num_matches.item() / len(self.val_loader.dataset),
        }

    def evaluate_and_logging_metrics(self):
        if self.training_args.eval_steps is None:
            return None
        if self.global_steps % self.training_args.eval_steps != 0:
            return None

        eval_metrics = self.evaluate()

        if self.accelerator.is_main_process:
            self.training_bar.write(
                f"Evalute: epoch {self.epoch} / {self.training_args.num_train_epochs}, "
                f"evaluation loss: {eval_metrics['loss']:.4f}, "
                f"accuracy@top1: {eval_metrics['accuracy@top1']}, "
                f"accuracy@top5: {eval_metrics['accuracy@top5']}"
            )
            self.accelerator.log(
                {
                    "evaluation loss": eval_metrics["loss"],
                    "accuracy@top1": eval_metrics["accuracy@top1"],
                    "accuracy@top5": eval_metrics["accuracy@top5"],
                },
                step=self.global_steps,
            )

    def maybe_load_checkpoint(self):
        self.global_steps = 0

        if self.training_args.resume_from_checkpoint is None:
            return

        checkpoint_path = self.training_args.resume_from_checkpoint

        # 如果是相对路径，则将组装为相对于 output_dir 的绝对路径
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(
                self.training_args.output_dir, checkpoint_path
            )

        if not os.path.exists(checkpoint_path):
            return

        # 如果路径中存在 checkpoint 的目录列表，则提取最后一个 checkpoint 目录
        checkpoints = sorted(
            [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if len(checkpoints) > 1:
            checkpoint_path = os.path.join(checkpoint_path, checkpoints[-1])

        self.accelerator.load_state(checkpoint_path)
        path = os.path.basename(checkpoint_path)
        training_difference = os.path.splitext(path)[0]
        self.global_steps = int(training_difference.replace("checkpoint-", ""))

    def save_checkpoints(self):
        if self.training_args.save_steps is None:
            return

        if self.global_steps % self.training_args.save_steps != 0:
            return

        checkpoint_dir = f"{self.training_args.output_dir}/checkpoints/checkpoint-{self.global_steps}"
        self.accelerator.save_state(checkpoint_dir, self.training_args.save_safetensors)

        if (
            self.training_args.save_total_limit is not None
            and self.accelerator.is_main_process
        ):
            # Limit total number of checkpoints
            checkpoints_dir = f"{self.training_args.output_dir}/checkpoints"
            checkpoints = sorted(
                [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[-1]),
            )
            # Keep only the most recent 3 checkpoints
            max_checkpoints = self.training_args.save_total_limit
            for checkpoint in checkpoints[:-max_checkpoints]:
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
                shutil.rmtree(checkpoint_path)

        self.accelerator.wait_for_everyone()


def main():
    # =========  prepare arguments parser and logger =============
    parser = simple_parsing.ArgumentParser(
        "Image Classification Trainer", add_config_path_arg=True
    )
    parser.add_arguments(TrainingArguments, dest="training")
    parser.add_arguments(ModelArguments, dest="model")
    parser.add_arguments(DatasetArguments, dest="dataset")
    args = parser.parse_args()

    training_args = args.training
    model_args = args.model
    dataset_args = args.dataset

    # ========= setup logging =========
    logging.basicConfig(
        format="[%(name)s][%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.seed is not None:
        set_seed(training_args.seed)

    # ========= build model =========
    model = create_model(model_args.name, pretrained=model_args.pretrained)

    # ========= build optimizer, scheduler, grad_scaler =========
    optimizer = optim.SGD(
        model.parameters(),
        lr=training_args.learning_rate,
        momentum=training_args.momentum,
        weight_decay=training_args.weight_decay,
    )
    lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # ========= build criterion =========
    criterion = nn.CrossEntropyLoss()
    # =========  prepare dataloader =============
    train_loader, val_loader = prepare_datalader(dataset_args, training_args)

    # ========= Build  Trainer ===========

    trainer = Trainer(
        model,
        training_args,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        lr_scheduler,
    )

    # ========= maybe loading checkpoint =============

    try:
        # start train
        trainer.train()
    except KeyboardInterrupt:
        logger.warning(
            f"The KeyboardInterrupt signal was captured, exiting all processes."
        )


if __name__ == "__main__":
    main()
