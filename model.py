import torch as th
import omegaconf as om
import pytorch_lightning as pl

from modules import *
from data import *
from utils import *


class VAD(pl.LightningModule):
    def __init__(self, hparams: om.DictConfig):
        super().__init__()
        self.hparams = hparams
        if not isinstance(hparams, om.DictConfig):
            hparams = om.DictConfig(hparams)
        self.hparams = om.OmegaConf.to_container(hparams, resolve=True)
        
        self.model = VADNet(**self.hparams['model'])
        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, x):
        probs = self.model(x)
        return probs

    def configure_optimizers(self):
        return th.optim.Adam(self.model.parameters() ,lr=self.hparams.training["lr"])

    def training_step(self, batch, batch_idx):
        x, t = batch['spectro'], batch['targets'].squeeze(1)
        probs = self.forward(x).squeeze(-1)
        loss = self.loss(probs, t)
        return loss

    def validation_step(self, batch, batch_idx):

        x, t = batch['spectro'], batch['targets'].squeeze(1)
        probs = self.forward(x).squeeze(-1)
        val_loss = self.loss(probs, t)

        ### COMPUTE metrics
        #eval_metrics = {
        #    "val_loss": val_loss,
        #    "eval_loss_recons": "TODO",
        #    "eval_contour_violation_ratio": "TODO",
        #}

        # self.log_dict(eval_metrics, on_step=False, on_epoch=True)

        return th.mean(val_loss)
