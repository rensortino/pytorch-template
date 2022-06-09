import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from datamodules.ddloader import DeviceDataLoader

class MNISTDataModule:
    def __init__(self, data_dir: str = 'data'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.num_classes = 10

        # Assign train/val datasets for use in dataloaders
        mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)

        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self, batch_size=16, device='cpu'):
        dl = DataLoader(self.mnist_train, batch_size=batch_size)
        return DeviceDataLoader(dl, device)

    def val_dataloader(self, batch_size=16, device='cpu'):
        dl = DataLoader(self.mnist_val, batch_size=batch_size)
        return DeviceDataLoader(dl, device)

    def test_dataloader(self, batch_size=16, device='cpu'):
        dl = DataLoader(self.mnist_test, batch_size=batch_size)
        return DeviceDataLoader(dl, device)