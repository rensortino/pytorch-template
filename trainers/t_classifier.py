from attrdict import AttrDict
from tqdm import tqdm
import torch.nn.functional as F
from models.m_classifier import Classifier
from trainers.t_base import BaseTrainer
from torch.optim import Adam

class ClsTrainer(BaseTrainer):

    def __init__(self, lr, resume=None, scheduler=None, **kwargs):

        '''
        kwargs:
            one_batch: trains the model on just one batch, for debugging purposes
            log_every_n_epochs: logging frequency
            log_weights: bool that sets whether to log weights as histograms
            log_gradients: bool that sets whether to log gradients as histograms
            batch_size: int defining the batch size
            debug: flag for debugging code 
        '''
        super().__init__(resume)
        self.kwargs = AttrDict(kwargs)
        self.model = Classifier(1, 28, 28, 10).to(kwargs.device)
        self.opt = Adam(self.model.parameters(), lr)
        self.scheduler = scheduler

        self.criterion = F.nll_loss

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