import os
from typing import Callable, Optional, Tuple, List
import cv2
import hydra
import pandas as pd

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import VisionDataset, ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A

AGE_BUCKETS = (
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
)

def convert_age_bucket_to_label(age_bucket: str) -> int:
    return AGE_BUCKETS.index(age_bucket)


class PandasDataset(VisionDataset):
    def __init__(
        self,
        root,
        path_to_csv_file,
        input_name,
        target_name,
        transform: Optional[A.Compose] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.dataset = pd.read_csv(path_to_csv_file)
        self.input_name = input_name
        self.target_name = target_name

    def __getitem__(self, idx):
        item = self.dataset.iloc[idx]
        filename = os.path.join(self.root, item[self.input_name])
        image = cv2.imread(filename)
        target = item[self.target_name]
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.dataset)


class FairFaceAge(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data",
        data_folder:str = "FairFace",
        margin: float = 0.25,
        num_classes: int = 9,
        filename_column: str = "file",
        age_column: str = "age",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        aug: Optional[List] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        assert margin in [0.25, 1.25], "margin must be 0.25 or 1.25"
        self.img_dir = os.path.join(
            data_dir, data_folder, "margin025" if margin == 0.25 else "margin125"
        )
        assert os.path.exists(self.img_dir), f"{self.img_dir} does not exist"

        # data transformations
        train_augs = []
        if aug is not None:
            train_augs = [hydra.utils.instantiate(t) for t in aug]
        val_transforms = [A.Normalize(0.5, 0.5), ToTensorV2()]
        self.val_transforms = A.Compose(val_transforms)
        self.train_transforms = A.Compose(
            train_augs + val_transforms,
        )
        self.train_csv = os.path.join(data_dir, data_folder, "fairface_label_train.csv")
        self.val_csv = os.path.join(data_dir, data_folder, "fairface_label_val.csv")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return self.hparams.num_classes

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = PandasDataset(self.img_dir, self.train_csv, self.hparams.filename_column, self.hparams.age_column, self.train_transforms, convert_age_bucket_to_label)
            valset = PandasDataset(self.img_dir, self.val_csv, self.hparams.filename_column, self.hparams.age_column, self.val_transforms, convert_age_bucket_to_label)
            self.data_train, self.data_val, self.data_test = trainset, valset, valset
              

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
