import torch
import torch.nn as nn
import pblm


class BiLSTM_A(pblm.PrebuiltLightningModule):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.hidden_size = 256
        self.num_layers = 10
        self.input_size = 2500
        self.num_classes = 128

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers,
                            batch_first=True, bidirectional=True)
        self.dense = nn.Linear(self.hidden_size*2, self.num_classes)

        self.lstm2 = nn.LSTM(self.num_classes, self.hidden_size, self.num_layers,
                             batch_first=True, bidirectional=True)
        self.dense2 = nn.Linear(self.hidden_size*2, self.input_size)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, -1)

        # Hidden State
        h0 = torch.zeros(self.num_layers*2,
                         x.shape[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2,
                         x.shape[0], self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dense(out[:, -1, :])
        out = out.reshape(out.shape[0], 1, -1)
        h1 = torch.zeros(self.num_layers*2,
                         out.shape[0], self.hidden_size).to(self.device)
        c1 = torch.zeros(self.num_layers*2,
                         out.shape[0], self.hidden_size).to(self.device)

        out, _ = self.lstm2(out, (h1, c1))
        out = self.dense2(out[:, -1, :])
        return out
