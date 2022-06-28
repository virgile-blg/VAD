import matplotlib.pyplot as plt
import argparse
import torch as th
import yaml
import os

from model import *


class VADPredictor(object):
    def __init__(self, ckpt_folder, device) -> None:
        super().__init__()
        self.ckpt_folder = ckpt_folder
        self.cfg = yaml.load(open(os.path.join(ckpt_folder, 'hparams.yml')), Loader=yaml.FullLoader)
        self.model = self.load_model()
        self.EPS = 1e-8

    def load_model(self):
        model = VAD(self.cfg)
        state_dict = th.load(os.path.join(self.ckpt_folder, 'last.ckpt'))['state_dict']
        model.load_state_dict(state_dict)
        return model

    def get_mel_spec(self, input_audio):
        audio, _ = torchaudio.load(input_audio)
        melspec_params = {
            'sample_rate':  self.cfg['data']['sr'],
            'n_fft' : self.cfg['data']['nfft'],
            'hop_length' :  self.cfg['data']['hop_length'],
            'n_mels' : self.cfg['data']['n_mels'],
        }
        melspec = torchaudio.transforms.MelSpectrogram(**melspec_params)
        mel = th.log(melspec(audio)+self.EPS)
        return mel

    def predict(self, audio_path, threshold):
        mel = self.get_mel_spec(audio_path)
        with th.no_grad():
            probs = th.sigmoid(self.model(mel.unsqueeze(0)))
            probs = probs[0, :, 0].detach().numpy()
        
        probs[probs >= threshold] = 1
        probs[probs < threshold] = 0

        return probs

    def plot_result(self, audio_path):
        mel = self.get_mel_spec(audio_path)
        probs = self.predict(audio_path)

        plt.plot(probs*40, color=('red'))
        plt.pcolormesh(mel[0].numpy())

        return probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='path to file to predict VAD')
    parser.add_argument('-p', '--plot_results', action='store_true', default=False, help='Plot spectrogram and model predictions')
    parser.add_argument('-c', '--ckpt_folder', default='./checkpoints/128_mels', help='Path to model checkpoint')
    args = parser.parse_args()
    
    predictor = VADPredictor(ckpt_folder=args.ckpt_folder, device='cpu')

    if not args.plot_results:
        probs = predictor.predict(audio_path=args.input_file)
    else: 
        probs = predictor.plot_results(audio_path=args.input_file)

    print(probs)
