import os
import sys
import json
import random
import logging
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR, LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as T

import numpy as np
from PIL import Image
from tqdm import tqdm
import simple_parsing
from timm import create_model

logger = logging.getLogger(__name__)


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


def load_imagenet_datasets(data_root):
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )
    train_data = ImageNetDataset(data_root, split="train", transform=train_transform)

    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]
    )
    val_data = ImageNetDataset(data_root, split="val", transform=val_transform)

    return {"train": train_data, "val": val_data}


def compute_topk_matches(logits, labels, k):
    _, preds = torch.topk(logits, k, dim=-1)
    return (preds == labels.unsqueeze(-1)).sum().item()


@torch.inference_mode()
def evaluate(model, val_loader, criterion, device):
    top1_num_matches = 0
    top5_num_matches = 0
    total_loss = 0.0

    prediction_bar = tqdm(
        desc="Evaluating",
        total=len(val_loader),
        leave=False,
        dynamic_ncols=True,
        unit="batch",
    )
    for batch_data in val_loader:
        input_tensors, labels = batch_data
        input_tensors = input_tensors.to(device=device)
        labels = labels.to(device=device)

        logits = model(input_tensors)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        top1_num_matches += compute_topk_matches(logits, labels, k=1)
        top5_num_matches += compute_topk_matches(logits, labels, k=5)

        prediction_bar.update()

    return {
        "loss": total_loss / len(val_loader),
        "accuracy@top1": top1_num_matches / len(val_loader.dataset),
        "accuracy@top5": top5_num_matches / len(val_loader.dataset),
    }


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
    learning_rate: float = 0.1
    train_batch_size_per_device: int = 8
    eval_batch_size_per_device: int = 8
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0
    momentum: float = 0.9
    num_train_epochs: int = 3
    logging_steps: int = 500
    eval_steps: int = None
    save_steps: int = None
    save_safetensors: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False


def save_checkpoints(
    global_steps, training_args, model, optimizer, lr_scheduler, loss, eval_metrics
):
    if training_args.save_steps is None or global_steps % training_args.save_steps != 0:
        return
    save_dir = Path(f"{training_args.output_dir}/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    training_states = {
        "global_steps": global_steps,
        "traning_loss": loss.item(),
    }
    if eval_metrics is not None:
        training_states["eval_metrics"] = eval_metrics

    with open(f"{save_dir}/training_states.json", "w") as state_file:
        state_file.write(json.dumps(training_states, indent=4))

    if training_args.save_safetensors:
        from safetensors.torch import save_file

        save_file(model.state_dict(), f"{save_dir}/model.safetensors")
    else:
        torch.save(model.state_dict(), f"{save_dir}/model.pt")

    torch.save(optimizer.state_dict(), f"{save_dir}/optimizer.pt")
    torch.save(lr_scheduler.state_dict(), f"{save_dir}/scheduler.pt")

    # 保存整个训练环境的随机数生成器的状态
    # ref: huggingface transformers(v4.46.3): trainer.py#L3153
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_states["cuda"] = torch.cuda.get_rng_state()
    torch.save(rng_states, f"{save_dir}/rng_state.pth")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    lr_scheduler: LRScheduler,
    training_args: TrainingArguments,
    device: torch.device,
) -> None:
    global_steps = 0
    total_steps = training_args.num_train_epochs * len(train_loader)
    writer = SummaryWriter(f"{training_args.output_dir}/tensorboard/")
    training_bar = tqdm(total=total_steps, dynamic_ncols=True, unit="step")
    training_bar.set_description("Training")

    for i in range(1, training_args.num_train_epochs + 1):
        for batch_data in train_loader:
            global_steps += 1

            input_tensors, labels = batch_data
            input_tensors = input_tensors.to(device=device)
            labels = labels.to(device=device)

            logits = model(input_tensors)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if global_steps % training_args.logging_steps == 0:
                training_bar.write(
                    f"Train: epoch {i} / {training_args.num_train_epochs}, "
                    f"steps: {global_steps} / {total_steps}, "
                    f"training loss: {loss.item():.4f}"
                )
                writer.add_scalar("training loss", loss.item(), global_steps)

            eval_metrics = None
            if (
                training_args.eval_steps is not None
                and global_steps % training_args.eval_steps == 0
            ):
                eval_metrics = evaluate(model, val_loader, criterion, device=device)

                training_bar.write(
                    f"Evalute: epoch {i} / {training_args.num_train_epochs}, "
                    f"evaluation loss: {eval_metrics['loss']}, "
                    f"accuracy@top1: {eval_metrics['accuracy@top1']}, "
                    f"accuracy@top5: {eval_metrics['accuracy@top5']}"
                )

                writer.add_scalar("evaluation loss", eval_metrics["loss"], global_steps)
                writer.add_scalar(
                    "accuracy@top1", eval_metrics["accuracy@top1"], global_steps
                )
                writer.add_scalar(
                    "accuracy@top5", eval_metrics["accuracy@top5"], global_steps
                )

            save_checkpoints(
                global_steps,
                training_args,
                model,
                optimizer,
                lr_scheduler,
                loss,
                eval_metrics,
            )
            training_bar.update()

        lr_scheduler.step()


def main():
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

    # Setup logging
    logging.basicConfig(
        format="[%(name)s][%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.ERROR,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build data
    datasets = load_imagenet_datasets(dataset_args.data_root)
    train_data = datasets["train"]
    val_data = datasets["val"]
    small_val_data = random_split(val_data, lengths=[0.2, 0.8])[0]

    train_loader = DataLoader(
        train_data,
        training_args.train_batch_size_per_device,
        shuffle=True,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    val_loader = DataLoader(
        small_val_data,
        training_args.eval_batch_size_per_device,
        shuffle=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    # build model
    model = create_model(model_args.name, pretrained=model_args.pretrained)
    model = model.to(device=device)

    # build optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=training_args.learning_rate,
        momentum=training_args.momentum,
        weight_decay=training_args.weight_decay,
    )
    lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # build criterion
    criterion = nn.CrossEntropyLoss()

    # start train
    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        lr_scheduler,
        training_args,
        device,
    )


if __name__ == "__main__":
    main()
