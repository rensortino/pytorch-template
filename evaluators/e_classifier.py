from pathlib import Path
from attrdict import AttrDict
import torch
from tqdm import tqdm
from logzero import logger as lz_logger
from torchmetrics.functional import accuracy
import torch.nn.functional as F

class Evaluator:

    def __init__(self, model, logger, resume, **kwargs):

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
        self.start_epoch = 0
        self.logger = logger

        self.criterion = F.nll_loss

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

    def epoch_end(self, epoch_loss, phase):
        lz_logger.info(f'{phase.upper()} Epoch {self.epoch}: {epoch_loss}')

    @torch.no_grad()
    def validate_one_epoch(self, val_loader):
        epoch_loss = 0
        epoch_acc = 0
        with tqdm(val_loader) as t:

            # Setup
            t.set_description(f'Validating Epoch {self.epoch}')
            self.model.eval()

            # Validation loop
            for batch in t:
                with torch.no_grad():
                    acc, loss = self.inference_step(batch)
                    epoch_loss += loss
                    epoch_acc += acc

        # Metric aggregation
        epoch_loss /= len(val_loader)
        epoch_acc /= len(val_loader)        

        self.logger.log_metric(f'val/loss', epoch_loss, self.epoch)
        self.logger.log_metric(f'val/acc', epoch_acc, self.epoch)

        if self.scheduler:
            self.scheduler.step(epoch_loss)

        # Final additional logging
        if (self.epoch + 1) % self.kwargs.log_every_n_epochs == 0:
            if epoch_loss < self.checkpoint['loss']:
                self.save_best(epoch_loss, self.checkpoint)        
            self.epoch_end(epoch_loss, 'val')

        return epoch_loss

    @torch.no_grad()
    def test_one_epoch(self, test_loader):
        epoch_loss = 0
        epoch_acc = 0
        with tqdm(test_loader) as t:

            # Setup
            t.set_description(f'Testing Epoch {self.epoch}')
            self.model.eval()

            # Testing loop
            for batch in t:
                with torch.no_grad():
                    acc, loss = self.inference_step(batch)
                    epoch_loss += loss
                    epoch_acc += acc

        epoch_loss /= len(test_loader)
        epoch_acc /= len(test_loader)    

        # Final additional logging
        self.logger.log_metric(f'test/loss', epoch_loss, self.epoch)
        self.epoch_end(epoch_loss, 'test')
        self.epoch_end(epoch_acc, 'test')

        return epoch_loss

    @torch.no_grad()
    def inference_step(self, batch):

        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        return acc, loss.item()

    def save_best(self, epoch_loss, checkpoint):
        # Save best model at min val loss
        checkpoint['loss'] = epoch_loss
        checkpoint['epoch'] = self.epoch
        checkpoint['model'] = self.model.state_dict()
        checkpoint['opt'] = self.opt.state_dict()
        self.logger.save_ckpt(checkpoint, 'best.pt')