import torch as th
import omegaconf as om
import torchmetrics as tm
import pytorch_lightning as pl

from modules import *
from data import *
from utils import *


class VAD(pl.LightningModule):
    def __init__(self, hparams: om.DictConfig):
        super().__init__()
        self.hparams.update(hparams)
        if not isinstance(hparams, om.DictConfig):
            hparams = om.DictConfig(hparams)
        self.hparams.update(om.OmegaConf.to_container(hparams, resolve=True))
        
        self.model = VADNet(**self.hparams['model'])
        self.loss = nn.BCEWithLogitsLoss()
        self.auroc = tm.AUROC(num_classes=1)
        self.acc = tm.Accuracy(threshold=0.5)
        self.f1 = tm.F1Score(threshold=0.5)

    def forward(self, x):
        probs = self.model(x)
        return probs

    def configure_optimizers(self):
        optim_type = self.hparams.training["optim"]
        assert  optim_type in ['Adam', 'SDG']
        
        if self.hparams.training["optim"] == 'Adam':
            return th.optim.Adam(self.model.parameters() ,lr=self.hparams.training["lr"], weight_decay=self.hparams.training["weight_decay"])
        else: 
            return th.optim.SGD(self.model.parameters() ,lr=self.hparams.training["lr"], weight_decay=self.hparams.training["weight_decay"])

    def training_step(self, batch, batch_idx):
        x, t = batch['spectro'], batch['targets'].squeeze(1)
        probs = self.forward(x).squeeze(-1)
        loss = self.loss(probs, t)
        self.log_dict({'train_loss':th.mean(loss)}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):

        x, t = batch['spectro'], batch['targets'].squeeze(1)
        probs = self.forward(x).squeeze(-1)
        val_loss = self.loss(probs, t)

        probs = probs.squeeze(0)
        t = t.int().squeeze(0)

        # Compute metrics
        eval_metrics = {
            "val_loss": th.mean(val_loss),
            "auroc": self.auroc(probs, t),
            "accuracy": self.acc(probs, t),
            "F1": self.f1(probs, t)
        }

        self.log_dict(eval_metrics, on_step=False, on_epoch=True)

        return th.mean(val_loss)
