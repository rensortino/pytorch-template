import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from datamodules.ddloader import DeviceDataLoader

class BaseDataModule:
    def __init__(self, data_dir: str = 'data', **kwargs):

        self.data_dir = data_dir
        self.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        self.num_workers = kwargs.pop('num_workers', 0)
        self.batch_size = kwargs.pop('batch_size', 16)

    def train_dataloader(self, device='cpu'):
        dl = DataLoader(self.dset_train, batch_size=self.batch_size)
        return DeviceDataLoader(dl, device)

    def val_dataloader(self, device='cpu'):
        dl = DataLoader(self.dset_val, batch_size=self.batch_size)
        return DeviceDataLoader(dl, device)

    def test_dataloader(self, device='cpu'):
        dl = DataLoader(self.dset_test, batch_size=self.batch_size)
        return DeviceDataLoader(dl, device)