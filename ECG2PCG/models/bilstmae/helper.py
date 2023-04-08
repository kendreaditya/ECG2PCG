import torch.nn as nn
import numpy as np
import torchaudio as ta


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_size = 256
        self.num_layers = 10
        self.input_size = 2500
        self.num_classes = 128

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        return x


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()


class Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
