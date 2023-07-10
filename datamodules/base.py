import torchvision.transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
import os

from datamodules.ddloader import DeviceDataLoader


class BaseDataModule:
    def __init__(self, data_dir: str = 'data', batch_size: int = 64, workers: int = 8, device: str = "cuda"):

        self.data_dir = Path(data_dir)
        self.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        self.batch_size = batch_size
        self.workers = min(workers, os.cpu_count())
        self.device = device

    def _create_dataloader(self, dset, is_train=True):
        dl = DataLoader(dset, batch_size=self.batch_size,
                        num_workers=self.workers, shuffle=is_train, drop_last=is_train)
        return DeviceDataLoader(dl, self.device)

    @property
    def train_loader(self):
        return self._train_loader
    
    @property
    def val_loader(self):
        return self._val_loader
    
    @property
    def test_loader(self):
        return self._test_loader