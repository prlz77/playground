from backbones import get_backbone
import pytorch_lightning as pl
import torch.nn.functional as F
import torch

class Classifier(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = get_backbone(hparams)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        features, y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        features, probs = self(x)
        # we currently return the accuracy as the validation_step/test_step is run on the IPU devices.
        # Outputs from the step functions are sent to the host device, where we calculate the metrics in
        # validation_epoch_end and test_epoch_end for the test_step.
        acc = self.accuracy(probs, y)
        loss = F.cross_entropy(probs, y)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        features, logits = self(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        # currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    # def validation_epoch_end(self, outputs) -> None:
    #     # since the training step/validation step and test step are run on the IPU device
    #     # we must log the average loss outside the step functions.
    #     self.log("val_acc", torch.stack(outputs["val_acc"]).mean(), prog_bar=True)
    #     self.log("val_loss", torch.stack(outputs["val_loss"]).mean(), prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        self.log("test_acc", torch.stack(outputs).mean())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
