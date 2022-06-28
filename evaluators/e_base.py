from pathlib import Path
from typing import OrderedDict
import torch
from logzero import logger as lz_logger

class BaseEvaluator:

    def __init__(self, logger, resume=None):

        self.start_epoch = 0
        self.logger = logger

        if resume:
            self.load_checkpoint(resume)
        else:
            self.checkpoint = {
                'epoch': 0,
                'loss': float('inf'),
                'model': OrderedDict() 
            }
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

    def epoch_end(self, epoch_loss, phase):
        lz_logger.info(f'{phase.upper()} Epoch {self.epoch}: {epoch_loss}')

    @torch.no_grad()
    def validate_one_epoch(self, val_loader):
       pass

    @torch.no_grad()
    def test_one_epoch(self, test_loader):
        pass

    @torch.no_grad()
    def inference_step(self, batch):
        pass

    def save_best(self, epoch_loss, checkpoint):
        # Save best model at min val loss
        checkpoint['loss'] = epoch_loss
        checkpoint['epoch'] = self.epoch
        checkpoint['model'] = self.model.state_dict()
        checkpoint['opt'] = self.opt.state_dict()
        self.logger.save_ckpt(checkpoint, 'best.pt')