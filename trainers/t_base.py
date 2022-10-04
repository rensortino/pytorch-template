from pathlib import Path
import torch
from logzero import logger as lz_logger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
class BaseTrainer:

    def __init__(self, resume, scheduler=None, device='cpu'):

        '''
        kwargs:
            one_batch: trains the model on just one batch, for debugging purposes
            log_every_n_epochs: logging frequency
            log_weights: bool that sets whether to log weights as histograms
            log_gradients: bool that sets whether to log gradients as histograms
            batch_size: int defining the batch size
            debug: flag for debugging code 
        '''

        self.device = device
        self.get_scheduler(scheduler)
        self.init_checkpoint(resume)
        self.start_epoch = 0


    def init_checkpoint(self, resume):
        if resume:
            self.load_checkpoint(resume)

        self.best_ckpt = {
                'epoch': 0,
                'loss': float('inf'),
            }

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.start_epoch = checkpoint['epoch']
        else:
            self.model.load_state_dict(checkpoint)

    def set_scheduler(self, scheduler):
        if not scheduler:
            self.scheduler = None
        if scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.opt, patience=50)

    def train_one_epoch(self, train_loader):
        pass

    def training_step(self, batch):
        pass

    def update_best_ckpt(self, epoch_loss, epoch):
        if epoch_loss < self.best_ckpt['loss']:
            # Save best model at min val loss
            self.best_ckpt['loss'] = epoch_loss
            self.best_ckpt['epoch'] = epoch
            self.best_ckpt['model'] = self.model.state_dict()
            self.best_ckpt['opt'] = self.opt.state_dict()