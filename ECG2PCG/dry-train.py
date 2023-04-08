import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wfdb
import torchvision
import torch.utils.data
from scipy import signal
from scipy.io import wavfile
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.functional import accuracy, precision, recall
from sklearn import preprocessing, metrics, model_selection
import pandas as pd
import torch
import wandb
import models
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
os.environ['WANDB_MODE'] = 'dryrun'
sys.path.append("./")
exec("from modules.data import Preprocessor")

capacity = 64
lolno = False


class PrebuiltLightningModule(pl.LightningModule):
    def __init__(self, name):
        super().__init__()

        # Metrics
        self.seed = np.random.randint(220)
        pl.seed_everything(seed=self.seed)

        # Loss
        self.criterion = nn.MSELoss()

        # Model Name
        self.name = name
        self.set_model_name()

        # Model Tags
        self.model_tags = [str(self.criterion), self.name]

    def set_model_name(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.model_name = f"{self.name}-{timestamp}"

    def configure_optimizers(self):
        optimzer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimzer

    def metrics_step(self, outputs, targets, loss, prefix=""):

        return {f'{prefix}loss': loss}

    # wandb table
    def log_step(self, metrics, **kwargs):
        for key in metrics:
            if "stats" not in key:
                self.log(
                    f"{key}",
                    metrics[key],
                    prog_bar=kwargs["prog_bar"],
                    on_step=kwargs["on_step"],
                    on_epoch=kwargs["on_epoch"])

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        # Logs metrics
        self.log("loss", loss, prog_bar=False,
                 on_step=True, on_epoch=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        if batch_idx == 1:
            plt.imshow(outputs[0][0].cpu())
            plt.show()

        return {"validation-loss": loss}

    # check metric average calculation
    def validation_epoch_end(self, outputs):
        avg_metrics = {key: 0.0 for key in outputs[0]}
        for n, metrics in enumerate(outputs):
            for key in metrics:
                if "stats" not in key:
                    avg_metrics[key] = (
                        (n)*avg_metrics[key]+metrics[key])/(n+1)

        self.log(key, avg_metrics[key], prog_bar=True,
                 on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.forward(data)
        loss = self.criterion(outputs, targets)

        return {"test-loss": loss}

    def test_epoch_end(self, outputs):
        avg_metrics = {key: 0.0 for key in outputs[0]}

        for n, metrics in enumerate(outputs):
            for key in metrics:
                avg_metrics[key] = (
                    (n)*avg_metrics[key]+metrics[key])/(n+1)

        self.log(key, avg_metrics[key], prog_bar=False,
                 on_step=False, on_epoch=True)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        i, target = dataset[idx]
        return int(target)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class Preprocessor():
    def __init__(self):
        pass

    def getAudioSignal(self, file, targetSamplingRate=500):
        sampleRate, data = wavfile.read(file)

        if sampleRate != targetSamplingRate:
            secs = len(data)/sampleRate
            num_samples = int(secs*targetSamplingRate)
            data = signal.resample(data, num_samples)

        return data

    def getFiles(self, dir, fileExtention="wav"):
        return [fn for fn in os.listdir(dir) if fileExtention in fn]

    def timeSegmentation(self, data, length, sampleRate=500, includeLast=False):
        length_samples = length*sampleRate
        segmented_data = []

        if includeLast:
            data_length = len(data)
        else:
            data_length = len(data)-length_samples

        for i in range(0, data_length, length_samples):
            segmented_data.append(data[i:i+length_samples])
        return segmented_data

    def standardization(self, data):
        return (data - torch.mean(data))/torch.std(data)

    def waveletDenoise(self, s, threshold=5, type='db10', level=4):
        coeffs = pywt.wavedec(s, type, level=level)

        # Applying threshold
        for x in range(len(coeffs)):
            coeffs[x] = pywt.threshold(coeffs[x], threshold, 'soft')

        # Reconstruing denoise signal (IDWT)
        reconstruction = pywt.waverec(coeffs, type)
        return reconstruction

    def savgolDenoise(self, data, window=10, order=None):
        return torch.from_numpy(signal.savogal_filter(data, window, order))

    def combineDatasets(self, dataset_path):
        data, labels = [], []
        for dir in dataset_path:
            dataset = torch.load(dir)
            data.append(dataset["data"])
            labels.append(dataset["labels"])

        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        return data, labels

    def toTensorDatasets(self, data, labels, split_ratio, **kwargs):
        data_splits = []
        labels_splits = []

        temp_data, temp_labels = data, labels

        for i in range(len(split_ratio)-1):
            splits = [split_ratio[i], sum(split_ratio[i+1:])]
            splits = [1/(sum(splits)/splits[0]), 1/(sum(splits)/splits[1])]

            x_split_1, x_split_2, y_split_1, y_split_2 = model_selection.train_test_split(
                temp_data, temp_labels, train_size=splits[0], test_size=split_ratio[1], shuffle=False)

            data_splits.append(x_split_1)
            labels_splits.append(y_split_1)

            if i == len(split_ratio)-2:
                data_splits.append(x_split_2)
                labels_splits.append(y_split_2)

            temp_data, temp_labels = x_split_2, y_split_2

        tensorDatasets = []

        for x, y in zip(data_splits, labels_splits):
            dataset = TensorDataset(x, y)
            tensorDatasets.append(dataset)

        return tensorDatasets

    def dataloaders(self, datasets, **kwargs):
        dataloaders = []
        for dataset in datasets:
            dataloaders.append(DataLoader(
                dataset, batch_size=kwargs['batch_size']))
        return dataloaders


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c,
                               kernel_size=4, stride=2, padding=1)  # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2,
                               kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c*3,
                               kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
        self.conv4 = nn.Conv2d(in_channels=c*3, out_channels=c*4,
                               kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.conv4 = nn.ConvTranspose2d(
            in_channels=c*4, out_channels=c*3, kernel_size=4, stride=2, padding=1)

        self.conv3 = nn.ConvTranspose2d(
            in_channels=c*3, out_channels=c*2, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = self.conv1(x)
        return x


class Autoencoder(PrebuiltLightningModule):
    def __init__(self):
        super(Autoencoder, self).__init__(__class__.__name__)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


os.environ['WANDB_MODE'] = 'dryrun'


class TrainerSetup():
    def __init__(self):

        # Model init
        model = Autoencoder()
        pp = Preprocessor()
        dataset = torch.load(
            "K:\\OneDrive - Cumberland Valley School District\\Education\\Activates\\Science Fair\\PCG-Science-Fair\\ECG2PCG\\data\\preprocessed\\physionet-a-spec.pt")

        datasets = pp.toTensorDatasets(
            dataset["ECG"].float(), dataset["PCG"].float(), [0.8, .1, 0.1])

        # del dataset
        # del labels

        train_dataloader, validation_dataloader, test_dataloader = pp.dataloaders(
            datasets, batch_size=32)

        del datasets

        # Checkpoints
        val_loss_cp = pl.callbacks.ModelCheckpoint(monitor='validation-loss')

        trainer = pl.Trainer(max_epochs=100, gpus=1, fast_dev_run=False,
                             auto_lr_find=False, auto_scale_batch_size=True, log_every_n_steps=1,
                             checkpoint_callback=val_loss_cp)

        # Train Model
        trainer.fit(model, train_dataloader, validation_dataloader)

        # Load best model with lowest validation
        self.model = model.load_from_checkpoint(
            val_loss_cp.best_model_path)

        # Test model on testing set
        self.results = trainer.test(model, test_dataloader)


trainerSetup = TrainerSetup()
