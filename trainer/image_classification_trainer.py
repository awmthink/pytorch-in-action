import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR, LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    RandomSampler,
    DistributedSampler,
)

import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
import simple_parsing
from timm import create_model

logger = logging.getLogger(__name__)


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def seed_everything(seed: Optional[int] = None):
    if seed is None:
        return

    if is_main_process():
        logger.info(f"Global seed set to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    resume_from_checkpoint: Optional[str] = None


class ImageNetDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", transform=None):
        valid_splits = ["train", "val"]
        if split not in valid_splits:
            raise ValueError(f"Invalid split: {split}. Choose from {valid_splits}.")

        self.split = split
        self.transform = transform
        self.image_root = os.path.join(data_root, split)

        # Load meta data
        meta_file_path = os.path.join(data_root, f"meta/{split}.txt")
        self.img_lst, self.labels = self._load_meta(meta_file_path)

    def _load_meta(self, meta_file_path):
        """Helper function to load image paths and labels from meta file."""
        with open(meta_file_path, "r", encoding="utf-8") as file:
            data = [line.strip().split(" ") for line in file.readlines()]

        img_lst, labels = zip(*data)
        labels = list(map(int, labels))
        return img_lst, labels

    def __getitem__(self, index):
        """Retrieve the image and its label for a given index."""
        image_path = os.path.join(self.image_root, self.img_lst[index])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.int64)
        return image, label

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.img_lst)


def prepare_datalader(dataset_args, training_args):
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )
    train_data = ImageNetDataset(
        dataset_args.data_root, split="train", transform=train_transform
    )

    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]
    )
    val_data = ImageNetDataset(
        dataset_args.data_root, split="val", transform=val_transform
    )

    if get_world_size() > 1:
        train_loader = DataLoader(
            train_data,
            training_args.train_batch_size_per_device,
            sampler=DistributedSampler(train_data),
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

        # val_data = random_split(val_data, lengths=[0.2, 0.8])[0]
        val_loader = DataLoader(
            val_data,
            training_args.eval_batch_size_per_device,
            sampler=DistributedSampler(val_data),
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_data,
            training_args.train_batch_size_per_device,
            shuffle=True,
            generator=torch.Generator(),
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

        # val_data = random_split(val_data, lengths=[0.2, 0.8])[0]
        val_loader = DataLoader(
            val_data,
            training_args.eval_batch_size_per_device,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
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
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        device: torch.device,
    ):
        self.model = model
        self.training_args = training_args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.use_amp = training_args.fp16 or training_args.bf16
        self.dtype = (
            torch.float16
            if training_args.fp16
            else torch.bfloat16 if training_args.bf16 else torch.float32
        )
        self.grad_scaler = torch.amp.GradScaler(enabled=self.use_amp)
        self.epoch = 0
        self.global_steps = 0

        self.total_steps = (
            self.training_args.num_train_epochs
            * len(self.train_loader)
            // self.training_args.grad_accumulation_batches
        )

        if is_main_process():
            self.writer = SummaryWriter(f"{self.training_args.output_dir}/tensorboard/")

    def train(self):
        if self.training_args.resume_from_checkpoint is not None:
            self.load_checkpoints()

        if get_world_size() > 1:
            self.model = DDP(self.model)

        batch_idx = self.global_steps * self.training_args.grad_accumulation_batches
        start_epoch = batch_idx // len(self.train_loader) + 1
        skip_batches = batch_idx % len(self.train_loader)

        if is_main_process():
            self.training_bar = tqdm(
                total=self.total_steps, dynamic_ncols=True, unit="step"
            )
            self.training_bar.set_description("Training")
            self.training_bar.update(self.global_steps)

        for epoch in range(start_epoch, self.training_args.num_train_epochs + 1):
            self.epoch = epoch
            # 用于保证每个 epoch 的 随机种子不一样
            self.set_dataloader_sampler_seed()

            for input_tensors, labels in self.train_loader:
                if skip_batches > 0:
                    # 这里可以优化，避免掉不必要的数据加载与数据预处理的耗时
                    skip_batches -= 1
                    continue

                non_blocking = self.training_args.dataloader_pin_memory
                input_tensors = input_tensors.to(
                    device=self.device, non_blocking=non_blocking
                )
                labels = labels.to(device=self.device, non_blocking=non_blocking)

                if batch_idx == 52:
                    print(f"rank = {dist.get_rank()}, data: ", labels[:10].tolist())

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=self.use_amp,
                ):
                    logits = self.model(input_tensors)
                    loss = self.criterion(logits, labels)
                    # 缩放损失并反向传播
                    self.grad_scaler.scale(loss).backward()

                batch_idx += 1

                # 未达到 AC 指定的 batch，则不进行梯度更新以及日志打印等
                if batch_idx % self.training_args.grad_accumulation_batches != 0:
                    continue

                # 进行梯度裁剪，防止出现梯度爆炸
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.training_args.max_grad_norm
                )
                # 更新模型参数
                self.grad_scaler.step(self.optimizer)
                # 更新 GradScaler
                self.grad_scaler.update()
                self.optimizer.zero_grad()
                self.global_steps += 1

                self.loging_training_metrics(loss)
                eval_metrics = self.evaluate_and_logging_metrics()
                self.save_checkpoints(loss, eval_metrics)

                if is_main_process():
                    self.training_bar.update()

            self.lr_scheduler.step()

    def set_dataloader_sampler_seed(self):
        if isinstance(self.train_loader.sampler, RandomSampler):
            self.train_loader.sampler.generator.manual_seed(self.epoch)
        elif isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.epoch)

    def loging_training_metrics(self, loss):
        if self.global_steps % self.training_args.logging_steps != 0:
            return
        if not is_main_process():
            return
        self.training_bar.write(
            f"Train: epoch {self.epoch} / {self.training_args.num_train_epochs}, "
            f"steps: {self.global_steps} / {self.training_bar.total}, "
            f"training loss: {loss.item():.4f}"
        )
        self.writer.add_scalar("training loss", loss.item(), self.global_steps)

    @torch.inference_mode()
    def evaluate(self):
        top1_num_matches = torch.tensor(0, device=self.device)
        top5_num_matches = torch.tensor(0, device=self.device)
        total_loss = torch.tensor(0.0, device=self.device)

        if is_main_process():
            prediction_bar = tqdm(
                desc="Evaluating",
                total=len(self.val_loader),
                leave=False,
                dynamic_ncols=True,
                unit="batch",
            )

        for batch_data in self.val_loader:
            input_tensors, labels = batch_data
            input_tensors = input_tensors.to(device=self.device)
            labels = labels.to(device=self.device)

            with torch.autocast(
                device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp
            ):
                logits = self.model(input_tensors)
                loss = self.criterion(logits, labels)

            total_loss += loss
            top1_num_matches += _compute_topk_matches(logits, labels, k=1)
            top5_num_matches += _compute_topk_matches(logits, labels, k=5)

            if is_main_process():
                prediction_bar.update()

        if get_world_size() > 1:
            dist.all_reduce(total_loss, dist.ReduceOp.AVG)
            dist.all_reduce(top1_num_matches)
            dist.all_reduce(top5_num_matches)

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

        if is_main_process():
            self.training_bar.write(
                f"Evalute: epoch {self.epoch} / {self.training_args.num_train_epochs}, "
                f"evaluation loss: {eval_metrics['loss']}, "
                f"accuracy@top1: {eval_metrics['accuracy@top1']}, "
                f"accuracy@top5: {eval_metrics['accuracy@top5']}"
            )

            self.writer.add_scalar(
                "evaluation loss", eval_metrics["loss"], self.global_steps
            )
            self.writer.add_scalar(
                "accuracy@top1", eval_metrics["accuracy@top1"], self.global_steps
            )
            self.writer.add_scalar(
                "accuracy@top5", eval_metrics["accuracy@top5"], self.global_steps
            )
        return eval_metrics

    def save_checkpoints(self, loss, eval_metrics):

        if self.training_args.save_steps is None:
            return

        if self.global_steps % self.training_args.save_steps != 0:
            return

        save_dir = Path(f"{self.training_args.output_dir}/checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)

        if is_main_process():

            # logger.info(f"Saving checkpoint to: {save_dir}")

            training_states = {
                "global_steps": self.global_steps,
                "traning_loss": loss.item(),
            }
            if eval_metrics is not None:
                training_states["eval_metrics"] = eval_metrics

            with open(f"{save_dir}/training_states.json", "w") as state_file:
                json.dump(training_states, state_file, indent=2)

            if isinstance(self.model, DDP):
                model = self.model.module
            else:
                model = self.model

            if self.training_args.save_safetensors:
                from safetensors.torch import save_file

                save_file(model.state_dict(), f"{save_dir}/model.safetensors")
            else:
                torch.save(model.state_dict(), f"{save_dir}/model.pt")

            torch.save(self.optimizer.state_dict(), f"{save_dir}/optimizer.pt")
            torch.save(self.lr_scheduler.state_dict(), f"{save_dir}/scheduler.pt")
            torch.save(self.grad_scaler.state_dict(), f"{save_dir}/grad_scaler.pt")

        # 保存整个训练环境的随机数生成器的状态
        # ref: huggingface transformers(v4.46.3): trainer.py#L3153
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.get_rng_state()

        rank_id_str = ""
        if get_world_size() > 1:
            rank_id_str = f"_{dist.get_rank()}"
        torch.save(rng_states, f"{save_dir}/rng_state{rank_id_str}.pth")

        if get_world_size() > 1:
            dist.barrier()

    def load_checkpoints(self):
        # 如果指定了 resume 的目录，但是目录不存在，则直接从零开始训练
        if not os.path.exists(self.training_args.resume_from_checkpoint):
            return 0

        if is_main_process():
            logger.info(
                f"loading checkpoint from: {self.training_args.resume_from_checkpoint}"
            )

        checkpoint_root = self.training_args.resume_from_checkpoint

        with open(f"{checkpoint_root}/training_states.json") as state_file:
            training_states = json.load(state_file)
            self.global_steps = training_states["global_steps"]

        if self.training_args.save_safetensors:
            from safetensors.torch import load_file

            model_states = load_file(
                f"{checkpoint_root}/model.safetensors", self.device.index
            )
            self.model.load_state_dict(model_states)
        else:
            model_states = torch.load(
                f"{checkpoint_root}/model.pt", self.device, weights_only=True
            )
            self.model.load_state_dict(model_states)

        optimizer_states = torch.load(
            f"{checkpoint_root}/optimizer.pt", self.device, weights_only=True
        )
        self.optimizer.load_state_dict(optimizer_states)

        scheduler_states = torch.load(
            f"{checkpoint_root}/scheduler.pt", self.device, weights_only=True
        )
        self.lr_scheduler.load_state_dict(scheduler_states)

        scaler_states = torch.load(
            f"{checkpoint_root}/grad_scaler.pt", self.device, weights_only=True
        )
        self.grad_scaler.load_state_dict(scaler_states)

        rank_id_str = ""
        if get_world_size() > 1:
            rank_id_str = f"_{dist.get_rank()}"

        rng_states = torch.load(
            f"{checkpoint_root}/rng_state{rank_id_str}.pth", "cpu", weights_only=False
        )
        random.setstate(rng_states["python"])
        np.random.set_state(rng_states["numpy"])
        torch.random.set_rng_state(rng_states["cpu"])

        if torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_states["cuda"], self.device)


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

    seed_everything(training_args.seed)

    # ========= setup logging =========
    logging.basicConfig(
        format="[%(name)s][%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ========= setup distiribted env =========
    if get_world_size() > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    # ========= build model =========
    model = create_model(model_args.name, pretrained=model_args.pretrained)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)

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
        device,
    )

    # ========= maybe loading checkpoint =============

    try:
        # start train
        trainer.train()
    except KeyboardInterrupt:
        logger.warning(
            f"The KeyboardInterrupt signal was captured, exiting all processes."
        )

    if get_world_size() > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
