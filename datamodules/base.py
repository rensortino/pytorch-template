import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from datamodules.ddloader import DeviceDataLoader

class BaseDataModule:
    def __init__(self, data_dir: str = 'data'):

        self.data_dir = data_dir
        self.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

    def train_dataloader(self, batch_size=16, device='cpu'):
        dl = DataLoader(self.dset_train, batch_size=batch_size)
        return DeviceDataLoader(dl, device)

    def val_dataloader(self, batch_size=16, device='cpu'):
        dl = DataLoader(self.dset_val, batch_size=batch_size)
        return DeviceDataLoader(dl, device)

    def test_dataloader(self, batch_size=16, device='cpu'):
        dl = DataLoader(self.dset_test, batch_size=batch_size)
        return DeviceDataLoader(dl, device)