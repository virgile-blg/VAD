import torch as th
import omegaconf as om
import pytorch_lightning as pl

from modules import *
from data import *
from utils import *
from losses import *


class VAD(pl.LightningModule):
    def __init__(self, hparams: om.DictConfig):
        super().__init__()
        self.hparams = hparams
        if not isinstance(hparams, om.DictConfig):
            hparams = om.DictConfig(hparams)
        self.hparams = om.OmegaConf.to_container(hparams, resolve=True)
        
        self.encoder = PianoGenieEncoder(
            rnn_dim=hparams.model["model_rnn_dim"],
            rnn_num_layers=hparams.model["model_rnn_num_layers"],
        )
        self.quantizer = IntegerQuantizer(hparams.model["num_buttons"])
        self.decoder = PianoGenieDecoder(
            rnn_dim=hparams.model["model_rnn_dim"],
            rnn_num_layers=hparams.model["model_rnn_num_layers"],
        )

        self.reconstruction_loss = ReconstructionLoss()
        self.margin_loss = MarginLoss()
        self.contour_loss = ContourLoss()

    def forward(self, k, t):
        e = self.encoder(k, t)
        b = self.quantizer(e)
        hat_k, _ = self.decoder(k, t, b)
        return hat_k, e

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.quantizer.parameters()) + list(self.decoder.parameters())
        return th.optim.Adam(params, lr=self.hparams.training["lr"])

    def training_step(self, batch, batch_idx):

        criteria = {}
        k, t = batch
        k = k.squeeze(1)
        t = t.squeeze(1)
        k_hat, e = self.forward(k, t)
        
        # Compute losses and update params
        loss_recons = self.reconstruction_loss(k, k_hat)
        loss_margin = self.margin_loss(e)
        loss_contour = self.contour_loss(k, e)

        # loss = torch.zeros_like(loss_recons)
        loss = loss_recons

        if self.hparams.training["loss_margin_multiplier"] > 0:
            loss += self.hparams.training["loss_margin_multiplier"] * loss_margin
        if self.hparams.training["loss_contour_multiplier"] > 0:
            loss += self.hparams.training["loss_contour_multiplier"] * loss_contour

        criteria['recons_loss'] = loss_recons
        criteria['margin_loss'] = loss_margin
        criteria['contour_loss'] = loss_contour
        criteria['train_loss'] = loss
        
        self.log_dict(criteria, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        eval_k, eval_t = batch
        eval_k = eval_k.squeeze(1)
        eval_t = eval_t.squeeze(1)
        eval_hat_k, eval_e = self.forward(eval_k, eval_t)

        eval_b = self.quantizer.real_to_discrete(eval_e)

        eval_loss_recons = F.cross_entropy(
            eval_hat_k.view(-1, PIANO_NUM_KEYS),
            eval_k.view(-1),
            reduction="none"
        )
        eval_violates = th.logical_not(
            th.sign(th.diff(eval_k, dim=1))
            == th.sign(th.diff(eval_b, dim=1)),
        ).float()
        
        eval_metrics = {
            "eval_loss_recons": th.mean(eval_loss_recons),
            "eval_contour_violation_ratio": th.mean(eval_violates),
        }

        self.log_dict(eval_metrics, on_step=False, on_epoch=True)

        return th.mean(eval_loss_recons)
