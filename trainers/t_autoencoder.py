from tqdm import tqdm
import torch.nn.functional as F
from trainers.t_base import BaseTrainer
from torch.optim import Adam
import torch
import wandb

class AETrainer(BaseTrainer):

    def __init__(self, model, **init_kwargs):
        '''
        Autoencoder for MNIST dataset
        '''
        super().__init__(**init_kwargs)
        self.model = model
        self.opt = Adam(self.model.parameters(), self.lr)
        self.criterion = F.l1_loss

    def training_step(self, batch, return_preds=False):
        x, _ = batch
        x_hat = self.model(x)
        loss = self.criterion(x, x_hat)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        loss_dict = {}
        loss_dict["rec_loss"] = loss.item()

        if return_preds:
            return loss_dict, x_hat

        return loss_dict

    def eval_step(self, batch):
        x, _ = batch
        with torch.no_grad():
            x_hat = self.model(x)
        loss = self.criterion(x_hat, x)

        metrics = {}
        metrics["rec_loss"] = loss.item()

        return metrics

    def visualize_results(self, epoch):
        sample_batch = next(iter(self.test_loader))
        x, _ = sample_batch
        x_hat = self.model(x)

        # Log images
        self.logger.save_image(x[:32], f"gt", epoch)
        self.logger.save_image(x_hat[:32], f"rec", epoch)
        wandb_gt = []
        wandb_rec = []
        for i in range(min(8, len(x))):
            wandb_gt.append(wandb.Image(x[i]))
            wandb_rec.append(wandb.Image(x_hat[i]))
        wandb.log({"Ground Truth": wandb_gt})
        wandb.log({"Reconstructed": wandb_rec})