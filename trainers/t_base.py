from pathlib import Path
import torch
from logzero import logger as lz_logger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.misc import instantiate_from_config
from utils.logger import Logger


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
class BaseTrainer:

    def __init__(self, resume, device='cpu', **kwargs):

        '''
        kwargs:
            log_freq: logging frequency
            batch_size: int defining the batch size
            debug: flag for debugging mode 
        '''

        self.device = device
        self.init_checkpoint(resume)
        self.start_epoch = 0
        self.num_epochs = kwargs.pop(kwargs["epochs"], 300)
        self.best_ckpt = {"loss": float("inf")}

        # Setup Logger
        out_dir = kwargs.pop("output_dir", "output")
        run_name = kwargs.pop("run_name", "placeholder")
        on_wandb = kwargs.pop("on_wandb", "False")
        self.logger = Logger(out_dir, run_name, on_wandb)

    def init_checkpoint(self, resume):
        if resume:
            self.load_checkpoint(resume)

    def load_checkpoint(self, ckpt_path):
        pass

    def train_one_epoch(self, train_loader):
        pass

    def training_step(self, batch):
        pass

    def update_best_ckpt(self, ckpt):
        if ckpt["loss"] < self.best_ckpt['loss']:
            self.best_ckpt = ckpt