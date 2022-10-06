from tqdm import tqdm
import torch.nn.functional as F
from models.m_classifier import Classifier
from trainers.t_base import BaseTrainer
from torch.optim import Adam

from utils.misc import instantiate_from_config

class ClsTrainer(BaseTrainer):

    def __init__(self, lr, resume=None, **kwargs):

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
        device = kwargs.pop('device', 'cuda')
        # self.model = Classifier(1, 28, 28, 10).to(device)
        model = kwargs.pop('model', None)
        scheduler = kwargs.pop('scheduler', None)
        self.model = instantiate_from_config(model).to(device)
        self.opt = Adam(self.model.parameters(), lr)
        self.criterion = F.nll_loss
        self.log_every_n_epochs = kwargs.pop('log_every_n_epochs', 15)
        self.scheduler = instantiate_from_config({'target': scheduler['target'], 'params': {'optimizer': self.opt, 'gamma': scheduler['params']['gamma']}}) if scheduler else None

    def train_one_epoch(self, dataloader):
        epoch_loss = 0
        self.model.train()
        for batch in tqdm(dataloader):
            loss = self.training_step(batch)
            epoch_loss += loss
        
        # Aggregating and logging metrics
        epoch_loss /= len(dataloader)
        return epoch_loss


    def training_step(self, batch):

        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def train_one_batch(self, batch):
        self.model.train()
        return self.training_step(batch)