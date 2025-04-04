import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter

from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt

            
    

DATALOADERS = {
    "alice": { # dataset for decoder-only transformer, demonstrating text generation
        "file": "datasets/alice_1.txt"
    },
    "shakespeare":{
        "file": "datasets/shakespeare.txt"
    },
    "pima": {
        "X": pd.read_csv("datasets/pima-indians-diabetes.csv").iloc[:, :-1].values,
        "y": pd.read_csv("datasets/pima-indians-diabetes.csv").iloc[:, -1].values,
    },
    "MNIST": {
        "train": datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        "test": datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
    },
    "FashionMNIST": {
        "train": datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        "test": datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
    },
    "CIFAR10": {
        "train": datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        "test": datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
    },
}


ACTIVATIONS = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Softmax": nn.Softmax(),
    "LeakyReLU": nn.LeakyReLU(),
    "PReLU": nn.PReLU(),
}


LAYERS = {
    "Flatten": lambda start_dim, end_dim: nn.Flatten(start_dim, end_dim), #  added unnecessary arguments to avoid weird params.lambda error
    "Linear": lambda i, o: nn.Linear(i, o),
    "Conv2D": lambda i, o, k_size: nn.Conv2d(i, o, k_size),  # i = input channels (1 --> grayscale, 3 --> RGB), o = output channels (number of filters), k_size = kernel size
    "Conv1D": lambda i, o, k_size: nn.Conv1d(i, o, k_size),
    "Conv3D": lambda i, o, k_size: nn.Conv3d(i, o, k_size),
    "MaxPool2D": lambda k_size, stride: nn.MaxPool2d(k_size, stride),
    "MaxPool1D": lambda k_size, stride: nn.MaxPool1d(k_size, stride),
    "MaxPool3D": lambda k_size, stride: nn.MaxPool3d(k_size, stride),
    "LSTM": lambda i, h_size: nn.LSTM(i, h_size),
    "GRU": lambda i, h_size: nn.GRU(i, h_size),
    "RNN": lambda i, h_size: nn.RNN(i, h_size),
    "Dropout": lambda p: nn.Dropout(p), # need to add functionality for dropout layer?
    "Decoder": lambda embed_dim, heads, hidden_dim: nn.TransformerDecoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=hidden_dim, batch_first=True),# USE IMPORTED CLASS FOR CONSTRUCTING DECODER HERE
    "Output": lambda p: nn.Dropout(p),
    # Output: [nn.Dropout(p), nn.Linear(embed_dim, vocab_size)], # would need to also access Linear layer after the dropout. Linear dimensions will be (embed_dim, vocab_size)
}


LOSSES = {
    "BCE": nn.BCELoss(),  # binary cross entropy --> 0 or 1 classification models
    "CrossEntropy": nn.CrossEntropyLoss(),  # multi-class classification models (including CNN)
    # "MSE": nn.MSELoss() # regression models
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),  # with logits for CNN binary classification
}

OPTIMIZERS = {
    "Adam": lambda model_params, lr: optim.Adam(
        model_params, lr
    ),  # momentum parameter is optional
    "AdamW": lambda model_params, lr: optim.AdamW(model_params, lr),
    "SGD": lambda model_params, lr: optim.SGD(model_params, lr),
    "RMSprop": lambda model_params, lr: optim.RMSprop(model_params, lr),
}
