import numpy as np
import os
from tqdm import tqdm
import wfdb
import torch
import sys
sys.path.append(".")
from modules.data import Preprocessor
import matplotlib.pyplot as plt

class Physionet():
    def __init__(self, input_size=2500):
        self.input_size = input_size
        self.data = {"PCG": [], "ECG": []}
        self.RECORDS = np.genfromtxt("./../data/training-a/RECORDS",
                                     delimiter="/n", dtype=str)
        self.PP = Preprocessor()

    def start(self):
        for record_name in self.RECORDS:
            record = wfdb.rdrecord(f'./../data/training-a/{record_name}')
            signal = np.transpose(record.p_signal)

            if signal.shape[0] != 2:
                continue

            for i in range(0, len(signal[0]), self.input_size):
                if signal[0][i:i+self.input_size].shape[-1] == self.input_size and np.isnan(signal.flatten()).any() == False:
                    pcg = [signal[0][i:i+self.input_size]]
                    ecg = [signal[1][i:i+self.input_size]]
                    pcg = self.PP.standardization(pcg)
                    ecg = self.PP.standardization(ecg)
                    self.data["PCG"].append(pcg)
                    self.data["ECG"].append(ecg)

        self.data["PCG"] = torch.tensor(self.data["PCG"])
        self.data["ECG"] = torch.tensor(self.data["ECG"])

        torch.save(self.data, "./../data/preprocessed/physionet-a.pt")


if __name__ == "__main__":
    physionet = Physionet()
    physionet.start()
