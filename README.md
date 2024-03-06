## Voice Activity Detector

This project implements a Voice Activity Detection algorithm based on the paper:  __Sofer, A., & Chazan, S. E. (2022). CNN self-attention voice activity detector. arXiv preprint arXiv:2203.02944.__

The `Data Processing` folder contains primarly a Notebook that has been used for the processing of the annoation and the data aumentation.
`data_utils.py` contains helper functions for annotation processing.
`energy_vad.py` contains code that was not written by myself. It is an implementaion of an energy-based VAD found on [GitHub](https://github.com/idnavid/py_vad_tool) that I used to extract noise signals from Librispeech samples.

The algorithm training pipeline is organised as follows:

- `data.py` implements the PyTorch dataset together with the Lightning DataModule
- `modules.py` implements the PyTorch neural network model
- `model.py` implements the Lightning Module for training
- `train.py` is the main script to start a training
- `inference.py` is a simple script to test the model on real-world audio
- `config` folder regroupe YAML file for experiment hyperparamters. The `baseline_sa_cnn.yml` is the hyperparameters set as described in the paper, while `128_mels.yml` is a slightly modified version.

You will also find some artifacts created after the training : 

- `checkpoints` folder is the saved model checkpoints, containing weights, optimizer state, hyperparams...
- `tb_logs` contains the Tensorboard logs
