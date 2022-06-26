from glob import glob
import scipy as sp
import torch as th
import pandas as pd
import torchaudio
import pytorch_lightning as pl
from typing import Optional
import numpy as np
import random
import os
import json


def get_frame_targets(audio_path, total_frames, hop_length, sr=16000):

    df = pd.read_csv(audio_path.replace('.wav', '.csv'))
    gt = th.zeros(total_frames)

    cur_frame = 0
    for i in df.index:
        utt_len = int(df.iloc[i].utt_time / (hop_length/sr)) # 10ms hop length

        gt[cur_frame:cur_frame+utt_len] = df.iloc[i].speech 
        cur_frame += utt_len

    return gt.unsqueeze(0)


class MelVADDataset(th.utils.data.Dataset):
    def __init__(self, path_list, n_frames=256, nfft=400, hop_length=160, n_mels=64, sr=16000, norm=False):
        self.path_list = path_list
        self.sr = sr
        self.mel_spec =  torchaudio.transforms.MelSpectrogram(n_fft=nfft, hop_length=hop_length, n_mels=n_mels)
        self.hop_length = hop_length
        self.n_frames = n_frames
        self.norm = norm
        # TODO
        #with open(os.path.join(data_dir, "stats.json"), "r") as f:
        #    self.stats = json.load(f)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        # Load track
        track_path = self.path_list[idx]
        audio, _ = torchaudio.load(track_path)
        # MelSpec
        spec = th.log(self.mel_spec(audio))
        
        # Get spec sample
        offset = int(th.randint(0, spec.shape[-1] - self.n_frames, [1]))
        print(offset, type(offset))
        sample = spec[:, :, offset:offset+self.n_frames]
        #print(offset, offset+self.n_frames)

        # Get targets
        targets = get_frame_targets(track_path, total_frames=spec.shape[-1], hop_length=self.hop_length)
        targets = targets[:, offset:offset+self.n_frames]
        #print(offset, offset+self.n_frames)


        if self.norm:
            # TODO
            streams = (streams - mean) / std
            return {"spectro": sample, "targets": targets, "mean": mean, "std": std}
        else:
            return {"spectro": sample, "targets": targets}


class VADMelDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, n_frames=256, nfft=400, hop_length=160, n_mels=64, sr=16000, norm=False,
                 n_workers=4, pin_memory=False, **kwargs):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_frames = n_frames
        self.nfft = nfft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sr = sr
        self.norm = norm
        self.n_workers = n_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        # Get train/val split
        path_list = glob(os.path.join(self.data_dir, '*.wav'))
        split = round(0.85*len(path_list))

        # Instantiate sub datasets
        self.train_set = MelVADDataset(path_list[:split], 
                                       n_frames=self.n_frames, 
                                       nfft=self.nfft, 
                                       hop_length=self.hop_length, 
                                       n_mels=self.n_mel, 
                                       sr=self.sr,
                                       norm=self.norm)
        self.val_set = MelVADDataset(path_list[split:], 
                                       n_frames=self.n_frames, 
                                       nfft=self.nfft, 
                                       hop_length=self.hop_length, 
                                       n_mels=self.n_mel, 
                                       sr=self.sr,
                                       norm=self.norm)

    def train_dataloader(self):
        return th.utils.data.DataLoader(self.train_set,
                                        batch_size=self.batch_size,
                                        pin_memory=self.pin_memory,
                                        shuffle=True,
                                        num_workers=self.n_workers)

    def val_dataloader(self):
        return th.utils.data.DataLoader(self.val_set,
                                        batch_size=1,
                                        pin_memory=False,
                                        shuffle=False,
                                        num_workers=self.n_workers)




if __name__ == "__main__":

    data_dir = "/data/upms-guitar"
    seed = 243
    subset = "gtr+acc-smc-clean-44100"
    mode = "train"
    n_samples = 130048

    # dataset = UPMSWAVDataset(data_dir, seed, subset, mode, n_samples)
    dataset = MonoUPMSWAVDataset(
        data_dir, seed, subset, mode, n_samples, "guitar", ["acc"]
    )
    print(len(dataset))
    print(dataset[0])
