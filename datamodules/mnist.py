from datamodules.base import BaseDataModule
import torchvision.transforms as T
from torch.utils.data import random_split
from torchvision.datasets import MNIST
class MNISTDataModule(BaseDataModule):
    def __init__(self, data_dir: str = 'data'):
        super().__init__(data_dir)
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.num_classes = 10

        # Assign train/val datasets for use in dataloaders
        mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)

        self.dset_train, self.dset_val = random_split(mnist_full, [55000, 5000])
        self.dset_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)