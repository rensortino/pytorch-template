from datamodules.base import BaseDataModule
import torchvision.transforms as T
from torch.utils.data import random_split
from torchvision.datasets import MNIST


class MNISTDataModule(BaseDataModule):
    def __init__(self, **init_kwargs):
        super().__init__(**init_kwargs)
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.num_classes = 10

        # Assign train/val datasets for use in dataloaders
        mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)

        self.train_dset, self.val_dset = random_split(mnist_full, [55000, 5000])
        self.test_dset = MNIST(self.data_dir, train=False, download=True, transform=self.transform)

        self._train_loader = self._create_dataloader(
            self.train_dset, is_train=True)
        self._val_loader = self._create_dataloader(
            self.val_dset, is_train=False)
        self._test_loader = self._create_dataloader(
            self.test_dset, is_train=False)