from tqdm import tqdm
import torch.nn.functional as F
from trainers.t_base import BaseTrainer
from torch.optim import Adam
import torch
import wandb
from torchvision.utils import save_image

class ClsTrainer(BaseTrainer):

    def __init__(self, model, **init_kwargs):
        '''
        Classifier for MNIST dataset
        '''
        super().__init__(**init_kwargs)
        self.model = model
        self.opt = Adam(self.model.parameters(), self.lr)
        self.criterion = F.nll_loss

    def training_step(self, batch, return_preds=False):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        loss_dict = {}
        loss_dict["cross_entropy"] = loss.item()

        if return_preds:
            preds = logits.argmax(dim=1)
            return loss_dict, preds

        return loss_dict

    def eval_step(self, batch):
        x, y = batch
        with torch.no_grad():
            logits = self.model(x)
        loss = self.criterion(logits, y)

        metrics = {}
        metrics["cross_entropy"] = loss.item()
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean()
        metrics['accuracy'] = accuracy.item()

        return metrics

    def visualize_results(self, epoch):
        sample_batch = next(iter(self.test_loader))
        x, y = sample_batch
        preds = self.model(x).argmax(dim=1)

        # Log images
        wandb_images = []
        for i in range(min(8, len(x))):
            self.logger.save_image(x[i], f"pred_{preds[i]}_label_{y[i]}.png", epoch)
            wandb_images.append(wandb.Image(x[i], caption=f"Pred: {preds[i]}, Label: {y[i]}"))
        wandb.log({"sample_preds": wandb_images})