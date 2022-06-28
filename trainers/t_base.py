from pathlib import Path
import torch
from logzero import logger as lz_logger
from torch import nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
class BaseTrainer:

    def __init__(self, logger, resume=None):

        '''
        kwargs:
            one_batch: trains the model on just one batch, for debugging purposes
            log_every_n_epochs: logging frequency
            log_weights: bool that sets whether to log weights as histograms
            log_gradients: bool that sets whether to log gradients as histograms
            batch_size: int defining the batch size
            debug: flag for debugging code 
        '''

        self.start_epoch = 0
        self.logger = logger

        if resume:
            self.load_checkpoint(resume)

        self.epoch = self.start_epoch

    def load_checkpoint(self, ckpt_path):
        
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
        self.checkpoint = torch.load(ckpt_path)
        if 'model' in self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model'])
        else:
            self.model.load_state_dict(self.checkpoint)
        
        if 'epoch' in self.checkpoint:
            self.start_epoch = self.checkpoint['epoch']

        if 'loss' not in self.checkpoint:
            self.checkpoint['loss'] = float('inf')

    def increment_epoch(self):
        self.epoch += 1

    def train_one_epoch(self, train_loader):
        pass

    def training_step(self, batch):
        pass

    def epoch_end(self, epoch_loss):

        lz_logger.info(f'Epoch {self.epoch}: {epoch_loss}')
       
        if not (self.kwargs.debug or self.kwargs.one_batch):
            ckpt = {
                'epoch': self.epoch,
                'model' : self.model.state_dict(),
                'opt' : self.opt.state_dict(),
            }
            self.logger.save_ckpt(ckpt)