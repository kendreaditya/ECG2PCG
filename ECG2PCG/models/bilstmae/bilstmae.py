import torch
import torch.nn as nn
import torchaudio as ta
import pblm


class Translatotron(pblm.PrebuiltLightningModule):
    def __init__(self, spectrogram=True, n_fft=400):
        super().__init__(self.__class__.__name__)
        self.spectrogram = spectrogram
        self.n_fft = n_fft

    def forward(self, x):
        if self.spectrogram:
            x = ta.transforms.Spectrogram(self.n_fft)

        return x
