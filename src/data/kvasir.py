import os
import torch
from PIL import Image
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms import v2

import pytorch_lightning as pl

from .prediction_dataset import PredictionDatasetWrapper

class KvasirDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        data_dir = os.path.abspath(os.path.expanduser(data_dir))
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.transform = transform
        
        self.images = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image, mask = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(image, mask)

        mask = (mask > 0).float() 
        
        return image, mask

class KvasirDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, seed: int = 0, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = v2.Compose([
            v2.Resize((256, 256)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        ])
        self.val_test_transform = v2.Compose([
            v2.Resize((256, 256)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        ])
        self.seed = seed

    def setup(self, stage: str):
        full_dataset = KvasirDataset(data_dir=self.data_dir)

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(self.seed)
        )

        train_indices, val_indices, test_indices = train_dataset.indices, val_dataset.indices, test_dataset.indices

        train_dataset = KvasirDataset(data_dir=self.data_dir, transform=self.train_transform)
        val_test_dataset = KvasirDataset(data_dir=self.data_dir, transform=self.val_test_transform)

        self.train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(val_test_dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(val_test_dataset, test_indices)
        self.predict_dataset = PredictionDatasetWrapper(self.test_dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=max(1, self.num_workers // 2))

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
