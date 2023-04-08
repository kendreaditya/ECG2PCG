import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
sys.path.append("./")
exec("from models import pblm")


capacity = 64


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c,
                               kernel_size=4, stride=2, padding=1)  # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2,
                               kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.conv2 = nn.ConvTranspose2d(
            in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv2(x))
        x = self.conv1(x)
        return x


class Autoencoder(pblm.PrebuiltLightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__(__class__.__name__)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    vae = Autoencoder()
    (vae.forward(torch.rand(1, 1, 48, 48)).shape)
