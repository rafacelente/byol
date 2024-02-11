import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision.models import resnet18
from torchmetrics import Accuracy

class ResNetModule(pl.LightningModule):
    def __init__(self):
        super(ResNetModule, self).__init__()
        resnet = resnet18(pretrained=False, num_classes=10)
        self.model = nn.Sequential(*list(resnet.children())[:-1])

    @classmethod
    def from_byol(cls, byol_module):
        resnet = cls()
        resnet.model.load_state_dict(byol_module.backbone.state_dict())
        return resnet

    def forward(self, x):
        return self.model(x).flatten(start_dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds  = torch.argmax(y_hat, dim=1)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        accuracy = Accuracy(task='multiclass', num_classes=10).to(self.device)
        acc = accuracy(preds, y)
        self.log('accuracy', acc)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer