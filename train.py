import yaml
import argparse
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import *
from data import *


def main(hparams_file):

    cfg = yaml.load(open(args.hparams), Loader=yaml.FullLoader)

    # Get checkpoint folder
    ckpt_folder = os.path.join('./checkpoints',Path(args.hparams).stem)
    os.makedirs(ckpt_folder, exist_ok=True)
    with open(os.path.join(ckpt_folder, 'hparams.yml'), 'w') as file:
        yaml.dump(cfg, file)

    # Load model
    model = VAD(cfg)
    # Load data 
    datamodule = VADMelDataModule(**cfg['data'])
    # TB Log
    logger = pl.loggers.TensorBoardLogger("tb_logs", name="VAD")
    # Callbacks
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_folder, **cfg['model_checkpoint'])
    # Trainer
    trainer = pl.Trainer(**cfg['trainer'],
                        logger=logger,
                        enable_checkpointing=checkpoint_callback,
                        callbacks=[checkpoint_callback, early_stop_callback])
    # Train
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--hparams", type=str, help="hparams config file")
    args = parser.parse_args()
    main(args.hparams)
