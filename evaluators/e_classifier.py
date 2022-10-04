import torch
from tqdm import tqdm
from torchmetrics.functional import accuracy

class ClsEvaluator:

    def __init__(self):
        pass

    @torch.no_grad()
    def validate_one_epoch(self, model, val_loader):
        epoch_loss = 0
        epoch_acc = 0

        # Setup
        model.eval()

        # Validation loop
        for batch in tqdm(val_loader):
            with torch.no_grad():
                acc, loss = self.inference_step(batch)
                epoch_loss += loss
                epoch_acc += acc

        # Metric aggregation
        epoch_loss /= len(val_loader)
        epoch_acc /= len(val_loader)        

        return epoch_loss

    @torch.no_grad()
    def test_one_epoch(self, model, test_loader):
        epoch_loss = 0
        epoch_acc = 0
        # Setup
        model.eval()

        # Testing loop
        for batch in tqdm(test_loader):
            with torch.no_grad():
                acc, loss = self.inference_step(batch)
                epoch_loss += loss
                epoch_acc += acc

        epoch_loss /= len(test_loader)
        epoch_acc /= len(test_loader)    

        return epoch_loss

    @torch.no_grad()
    def inference_step(self, batch):

        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        return acc, loss.item()