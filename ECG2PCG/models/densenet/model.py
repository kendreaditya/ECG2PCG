import Densenet as dn
import torch
import sys
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
sys.path.append("./")
exec("from models import pblm as pblm")


class Densenet(pblm.PrebuiltLightningModule):
    def __init__(self, input_size=(1, 1, 51, 51), in_channel=1, growth_rate=12, block_config=(6, 12, 24), channel_num=32, bn_size=4, dropout_rate=0):
        super().__init__(__class__.__name__)
        self.encoder = dn.DenseNet(growth_rate=growth_rate, block_config=block_config, num_init_features=channel_num,
                                   bn_size=bn_size, drop_rate=dropout_rate, memory_efficient=False)

        self.decoder = dn.DenseNet(
            growth_rate=growth_rate, block_config=block_config[::-1], num_init_features=channel_num, bn_size=4, memory_efficient=False)

    def forward(self, x):
        return x


if __name__ == "__main__":
    net = Densenet()
    encodeShape = net.encoder(torch.rand(1, 1, 51, 51)).shape
    print(encodeShape)

    print(net.decoder(torch.rand(encodeShape)).shape)
