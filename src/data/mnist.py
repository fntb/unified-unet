import torch
import torchvision
import pytorch_lightning as pl

class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, seed: int = 0, num_workers: int = 1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(degrees=10),
            torchvision.transforms.Pad(padding=2),
            torchvision.transforms.RandomCrop(28),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            # torchvision.transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.2, 2)),
        ])
        self.val_test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.seed = seed

    def prepare_data(self) -> None:
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            base_mnist_dataset = torchvision.datasets.MNIST(
                self.data_dir,
                train=True,
            )

            train_dataset, val_dataset = torch.utils.data.random_split(
                base_mnist_dataset,
                [0.9, 0.1],
                generator=torch.Generator().manual_seed(self.seed)
            )

            train_indices = train_dataset.indices
            val_indices = val_dataset.indices

            full_train_dataset = torchvision.datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.train_transform
            )

            full_val_dataset = torchvision.datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.val_test_transform
            )

            self.train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
            self.val_dataset = torch.utils.data.Subset(full_val_dataset, val_indices)

        elif stage == "test":
            self.test_dataset = torchvision.datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.val_test_transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def teardown(self, stage: str):
        ...
