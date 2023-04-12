import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from pathlib import Path

from datamodules.ddloader import DeviceDataLoader

class BaseDataModule:
    def __init__(self, data_dir: str = 'data', **kwargs):

        self.data_dir = Path(data_dir)
        self.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        self.batch_size = kwargs.pop('batch_size', 16)
        self.max_workers = min(self.batch_size, os.cpu_count())


    def train_dataloader(self, device='cpu', batch_size=None):
        bs = batch_size if batch_size else self.batch_size
        dl = DataLoader(self.dset_train, batch_size=bs)
        return DeviceDataLoader(dl, device)

    def val_dataloader(self, device='cpu', batch_size=None):
        bs = batch_size if batch_size else self.batch_size
        dl = DataLoader(self.dset_val, batch_size=bs)
        return DeviceDataLoader(dl, device)

    def test_dataloader(self, device='cpu', batch_size=None):
        bs = batch_size if batch_size else self.batch_size
        dl = DataLoader(self.dset_test, batch_size=bs)
        return DeviceDataLoader(dl, device)