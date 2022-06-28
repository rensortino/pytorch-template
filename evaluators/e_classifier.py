from attrdict import AttrDict
from evaluators.e_base import BaseEvaluator
import torch
from tqdm import tqdm
from torchmetrics.functional import accuracy
import torch.nn.functional as F

class ClsEvaluator(BaseEvaluator):

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

        super().__init__(logger, resume)
        self.kwargs = AttrDict(kwargs)
        self.model = model
        self.criterion = F.nll_loss

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