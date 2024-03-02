from byol.models.heads import BYOLPredictionHead, BYOLProjectionHead
from byol.utils import cosine_scheduler, update_momentum, get_default_byol_hparams
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchvision.models as models
from typing import Optional, Dict, Literal
from collections import OrderedDict


class BYOL(pl.LightningModule):
    def __init__(
            self, 
            hparams: Optional[Dict]=None, 
            model: Optional[nn.Module]=None,
            batch_type: Literal["tuple", "dict"]="tuple"
        ):
        super().__init__()
        hparams = get_default_byol_hparams() if hparams is None else hparams
        self.hparams.update(hparams)

        # Batch type depends on the dataloader
        self.batch_type = True if batch_type == "tuple" else False

        if model is None:
            resnet = models.resnet18(pretrained=False, num_classes=10)
            # In the models.resnet18, the last layer is a fully connected layer with 512 input features and 10 output features
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.target_backbone = nn.Sequential(*list(resnet.children())[:-1]) # -1 because we don't fc layers
        else:
            # In the unet model, the last layer a sequential layer with a avgpool and fc layer
            self.backbone = nn.Sequential()
            self.target_backbone = nn.Sequential()
            for name, module in list(model.named_children()):
                self.backbone.add_module(name, module)
                self.target_backbone.add_module(name, module)
            self.backbone.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))
            self.backbone.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))

        # Online network
        self.projection_head = BYOLProjectionHead(
            self.hparams["input_dim"],
            self.hparams["hidden_dim"],
            self.hparams["projection_dim"],
        )
        self.prediction_head = BYOLPredictionHead(
            self.hparams["projection_dim"],
            self.hparams["hidden_dim"],
            self.hparams["projection_dim"],
        )

        # Target Network
        self.target_projection_head = BYOLProjectionHead(
            self.hparams["input_dim"],
            self.hparams["hidden_dim"],
            self.hparams["projection_dim"],
        )

        # deactivate gradients for target network
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projection_head.parameters():
            param.requires_grad = False

        self.loss = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z) # q_theta(z_theta)
        return p
    
    def forward_target(self, x):
        y = self.target_backbone(x).flatten(start_dim=1)
        z = self.target_projection_head(y)
        return z.detach()
    
    def training_step(self, batch, batch_idx):
        momentum = cosine_scheduler(
            self.current_epoch,
            self.hparams["max_epochs"],
            1,
            0.996,
        )
        # update target network
        update_momentum(self.backbone, self.target_backbone, momentum)
        update_momentum(self.projection_head, self.target_projection_head, momentum)

        # FIXME: This is a temporary fix for the batch type
        if self.batch_type:
            x1, x2 = batch[0]
        else:
            x1, x2 = batch[0].float(), batch[1].float()
            

        p_theta_1 = self.forward(x1)
        z_epsilon_1 = self.forward_target(x1)
        p_theta_2 = self.forward(x2)
        z_epsilon_2 = self.forward_target(x2)

        loss = -0.5 * (self.loss(p_theta_1, z_epsilon_2) + self.loss(p_theta_2, z_epsilon_1)).mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        return self.backbone(x).flatten(start_dim=1)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.06)