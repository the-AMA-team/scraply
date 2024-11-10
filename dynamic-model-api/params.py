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


# UPDATE THIS FUNCTION WITH NEW OPTIONS + MAKE SURE LAST ACTIVATION CAN ONLY BE SIGMOID OR SOFTMAX
def are_params_valid(data):
    if data["input"] not in DATALOADERS.keys():
        return False

    for l in data["layers"]:
        if l["kind"] == "Linear":
            if not isinstance(l["args"], list) or len(l["args"]) != 2:
                return False

        elif l["kind"] in ["Conv1D", "Conv2D", "Conv3D"]:
            if not isinstance(l["args"], list) or len(l["args"]) != 3:
                print("invalid conv")
                return False

        elif l["kind"] in ["LSTM", "GRU", "RNN"]:
            if not isinstance(l["args"], list) or len(l["args"]) != 2:
                print("invalid rnn")
                return False

        if l["kind"] not in LAYERS.keys() and l["kind"] not in ACTIVATIONS.keys():
            print(l["kind"])
            print("invalid kind")
            return False

    if data["loss"] not in LOSSES.keys():
        print("invalid loss")
        return False
    if data["optimizer"]["kind"] not in OPTIMIZERS.keys():
        print("invalid optimizer")
        return False
    if not isinstance(data["epoch"], int):
        print("invalid epoch")
        return False
    if not isinstance(data["batch_size"], int):
        print("invalid batch_size")
        return False
    return True


# transform = transform (then transform = transforms.Compose([transforms.ToTensor(), transform]))
# ex: datasets.MNIST(root="data", train=True, download=true, transform=transform)

DATALOADERS = {
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
    "Flatten": nn.Flatten(), # no argumnets for Flatten()
    "Linear": lambda i, o: nn.Linear(i, o),
    "Conv2D": lambda i, o, k_size: nn.Conv2d(i, o, k_size), # i = input channels (1 --> grayscale, 3 --> RGB), o = output channels (number of filters), k_size = kernel size
    "Conv1D": lambda i, o, k_size: nn.Conv1d(i, o, k_size),
    "Conv3D": lambda i, o, k_size: nn.Conv3d(i, o, k_size),
    "LSTM": lambda i, h_size: nn.LSTM(i, h_size),
    "GRU": lambda i, h_size: nn.GRU(i, h_size),
    "RNN": lambda i, h_size: nn.RNN(i, h_size),
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
