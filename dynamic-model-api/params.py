import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ex: datasets.MNIST(root="data", train=True, download=true, transform=transform)

DATALOADERS = {
    "alice": { # dataset for decoder-only transformer, demonstrating text generation
        "text": open('datasets/alice_1.txt', 'r', encoding='utf-8').read() # FILE DOESNT CLOSE! --> could cause some issues
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
    "Flatten": nn.Flatten(),
    "Linear": lambda i, o: nn.Linear(i, o),
    "Conv2D": lambda i, o, k_size: nn.Conv2d(
        i, o, k_size
    ),  # i = input channels (1 --> grayscale, 3 --> RGB), o = output channels (number of filters), k_size = kernel size
    "Conv1D": lambda i, o, k_size: nn.Conv1d(i, o, k_size),
    "Conv3D": lambda i, o, k_size: nn.Conv3d(i, o, k_size),
    "LSTM": lambda i, h_size: nn.LSTM(i, h_size),
    "GRU": lambda i, h_size: nn.GRU(i, h_size),
    "RNN": lambda i, h_size: nn.RNN(i, h_size),
    "Dropout": lambda p: nn.Dropout(p),
    # need to add functionality for dropout layer
}


LOSSES = {
    "BCE": nn.BCELoss(),  # binary cross entropy --> 0 or 1 classification models
    "CrossEntropy": nn.CrossEntropyLoss(),  # multi-class classification models
    # "MSE": nn.MSELoss() # regression models
}


OPTIMIZERS = {
    "Adam": lambda model_params, lr: optim.Adam(
        model_params, lr
    ),  # momentum parameter is optional
    "AdamW": lambda model_params, lr: optim.AdamW(model_params, lr),
    "SGD": lambda model_params, lr: optim.SGD(model_params, lr),
    "RMSprop": lambda model_params, lr: optim.RMSprop(model_params, lr),
}
