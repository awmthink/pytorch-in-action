import os

import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import random_split

import pytorch_lightning as pl

from PIL import Image


normlization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normlization,
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normlization,
        ]
    ),
}


class ImageNetDataSet(data.Dataset):
    def __init__(
        self,
        root_dir,
        img_lst,
        labels,
        transform=transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.ToTensor()]
        ),
    ):
        self.root_dir = root_dir
        self.img_lst = img_lst
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_dir, self.img_lst[index])).convert(
            "RGB"
        )
        image = self.transform(image)
        return image, torch.tensor(self.labels[index], dtype=torch.int64)

    def __len__(self):
        return len(self.img_lst)


def _read_image_list(meta_file):
    with open(meta_file, "r") as file:
        img_lst, labels = zip(*(line.strip().split(" ") for line in file))
    labels = list(map(int, labels))
    return img_lst, labels


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    # Download data on single GPU
    def prepare_data(self):
        pass

    def setup(self, stage):
        train_lst, train_labels = _read_image_list(
            os.path.join(self.root_dir, "meta/train.txt")
        )
        self.train_ds = ImageNetDataSet(
            os.path.join(self.root_dir, "train"),
            train_lst,
            train_labels,
            transform["train"],
        )
        val_lst, val_labels = _read_image_list(
            os.path.join(self.root_dir, "meta/val.txt")
        )
        validation_ds = ImageNetDataSet(
            os.path.join(self.root_dir, "val"),
            val_lst,
            val_labels,
            transform["test"],
        )
        self.val_ds, self.test_ds = random_split(validation_ds, [30000, 20000])

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
