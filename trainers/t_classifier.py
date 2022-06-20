from pathlib import Path
from attrdict import AttrDict
import torch
from tqdm import tqdm
from logzero import logger as lz_logger
import torch.nn.functional as F

class Trainer:

    def __init__(self, model, opt, logger, resume=None, scheduler=None, **kwargs):

        '''
        kwargs:
            one_batch: trains the model on just one batch, for debugging purposes
            log_every_n_epochs: logging frequency
            log_weights: bool that sets whether to log weights as histograms
            log_gradients: bool that sets whether to log gradients as histograms
            batch_size: int defining the batch size
            debug: flag for debugging code 
        '''

        self.kwargs = AttrDict(kwargs)
        self.model = model
        self.opt = opt
        self.start_epoch = 0
        self.logger = logger
        self.scheduler = scheduler

        self.criterion = F.nll_loss

        if resume:
            self.load_checkpoint(resume)
        else:
            self.checkpoint = {
                'epoch': 0,
                'loss': float('inf'),
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

    def train_one_epoch(self, train_loader):
        epoch_loss = 0
        with tqdm(train_loader, disable=self.kwargs.one_batch) as t:

            # Setup
            self.model.train()
            t.set_description(f'Training Epoch {self.epoch}')

            # Training loop
            for i, batch in enumerate(t):
                loss = self.training_step(batch)
                epoch_loss += loss
                if self.kwargs.one_batch:
                    break
        
        # Aggregating and logging metrics
        epoch_loss /= len(train_loader)
        self.logger.log_metric(f'train/loss', epoch_loss, self.epoch)
            
        # Additional logging at the end of the epoch (e.g. checkpoints)
        if (self.epoch + 1) % self.kwargs.log_every_n_epochs == 0:
            self.epoch_end(epoch_loss, 'train')

        return epoch_loss


    def training_step(self, batch):

        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def epoch_end(self, epoch_loss):

        lz_logger.info(f'Epoch {self.epoch}: {epoch_loss}')
       
        if not (self.kwargs.debug or self.kwargs.one_batch):
            ckpt = {
                'epoch': self.epoch,
                'model' : self.model.state_dict(),
                'opt' : self.opt.state_dict(),
            }
            self.logger.save_ckpt(ckpt)

    def save_best(self, epoch_loss, checkpoint):
        # Save best model at min val loss
        checkpoint['loss'] = epoch_loss
        checkpoint['epoch'] = self.epoch
        checkpoint['model'] = self.model.state_dict()
        checkpoint['opt'] = self.opt.state_dict()
        self.logger.save_ckpt(checkpoint, 'best.pt')