from backbones import get_backbone
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from .classifier import Classifier

class DoubleClassifier(Classifier):
    def training_step(self, batch, batch_idx):
        x, y = batch
        features, y_hat = self(x)
        y1, y2 = torch.chunk(y_hat, 2, -1)
        loss = F.cross_entropy(y1, y // 10)
        loss += F.cross_entropy(y2, y % 10)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features, probs = self(x)
        y1, y2 = torch.chunk(probs, 2, -1)
        # we currently return the accuracy as the validation_step/test_step is run on the IPU devices.
        # Outputs from the step functions are sent to the host device, where we calculate the metrics in
        # validation_epoch_end and test_epoch_end for the test_step.
        acc = 0.5 * self.accuracy(y1, y // 10)
        acc += 0.5 * self.accuracy(y2, y // 10)
        loss = 0.5 * F.cross_entropy(y1, y // 10)
        loss += 0.5 * F.cross_entropy(y2, y % 10)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}
