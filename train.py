import yaml
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import *
from data import *


def main(hparams_file):

    cfg = yaml.load(open(args.hparams), Loader=yaml.FullLoader)

    model = VAD(cfg)

    logger = pl.loggers.TensorBoardLogger("tb_logs", name="VAD")

    checkpoint_callback = ModelCheckpoint(**cfg['model_checkpoint'])
    
    datamodule = VADMelDataModule(**cfg['data'])

    trainer = pl.Trainer(**cfg['trainer'],
                        logger=logger,
                        checkpoint_callback=checkpoint_callback,
                        callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)
