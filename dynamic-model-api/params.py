import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt


_DATA_DIR = Path(__file__).resolve().parent
_TORCHVISION_ROOT = _DATA_DIR / "data"
_PIMA_CSV = _DATA_DIR / "datasets" / "pima-indians-diabetes.csv"
_IMAGE_TRANSFORM = transforms.Compose([transforms.ToTensor()])
_DATASET_CACHE = {}


def _load_image_dataset(dataset_cls):
    return {
        "train": dataset_cls(
            root=str(_TORCHVISION_ROOT),
            train=True,
            download=True,
            transform=_IMAGE_TRANSFORM,
        ),
        "test": dataset_cls(
            root=str(_TORCHVISION_ROOT),
            train=False,
            download=True,
            transform=_IMAGE_TRANSFORM,
        ),
    }


def _load_dataset(name: str):
    if name == "alice":
        return {"file": str(_DATA_DIR / "datasets" / "alice_1.txt")}
    if name == "shakespeare":
        return {"file": str(_DATA_DIR / "datasets" / "shakespeare.txt")}
    if name == "pima":
        data = pd.read_csv(_PIMA_CSV, header=None).values
        return {"X": data[:, :-1], "y": data[:, -1]}
    if name == "MNIST":
        return _load_image_dataset(datasets.MNIST)
    if name == "FashionMNIST":
        return _load_image_dataset(datasets.FashionMNIST)
    if name == "CIFAR10":
        return _load_image_dataset(datasets.CIFAR10)
    raise KeyError(f"Unknown dataset: {name}")


def get_dataloader(name: str):
    """Load a dataset on first use and reuse it for later training jobs."""
    if name not in _DATASET_CACHE:
        _DATASET_CACHE[name] = _load_dataset(name)
    return _DATASET_CACHE[name]


ACTIVATIONS = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Softmax": nn.Softmax(dim=1), # forcing dim=1, as this is usually the case. 
    "LeakyReLU": nn.LeakyReLU(),
    "PReLU": nn.PReLU(),
    "No Activation": nn.Identity(), # does nothing
}


LAYERS = {
    "Flatten": lambda start_dim, end_dim: nn.Flatten(start_dim, end_dim),  #  added unnecessary arguments to avoid weird params.lambda error
    "Linear": lambda i, o: nn.Linear(i, o),
    "Conv1D": lambda _dim, i, o, k_size, stride, padding: nn.Conv1d(i, o, k_size, stride, padding),
    "Conv2D": lambda _dim, i, o, k_size, stride, padding: nn.Conv2d(i, o, k_size, stride, padding),  # i = input channels (1 --> grayscale, 3 --> RGB), o = output channels (number of filters), k_size = kernel size
    "Conv3D": lambda _dim, i, o, k_size, stride, padding: nn.Conv3d(i, o, k_size, stride, padding),
    "MaxPool1D": lambda _dim, k_size, stride, padding: nn.MaxPool1d(k_size, stride, padding),
    "MaxPool2D": lambda _dim, k_size, stride, padding: nn.MaxPool2d(k_size, stride, padding),
    "MaxPool3D": lambda _dim, k_size, stride, padding: nn.MaxPool3d(k_size, stride, padding),
    "AvgPool1D": lambda _dim, k_size, stride, padding: nn.AvgPool1d(k_size, stride, padding),
    "AvgPool2D": lambda _dim, k_size, stride, padding: nn.AvgPool2d(k_size, stride, padding),
    "AvgPool3D": lambda _dim, k_size, stride, padding: nn.AvgPool3d(k_size, stride, padding),
    "LSTM": lambda i, h_size, dropout: nn.LSTM(i, h_size, dropout),
    "GRU": lambda i, h_size, dropout: nn.GRU(i, h_size, dropout),
    "RNN": lambda i, h_size, dropout: nn.RNN(i, h_size, dropout),
    "Dropout": lambda p: nn.Dropout(p),  # need to add functionality for dropout layer?
    "Decoder": lambda embed_dim, heads, hidden_dim: nn.TransformerDecoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=hidden_dim, batch_first=True),  # USE IMPORTED CLASS FOR CONSTRUCTING DECODER HERE
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
    "Adam": lambda model_params, lr: optim.Adam(model_params, lr),  # momentum parameter is optional
    "AdamW": lambda model_params, lr: optim.AdamW(model_params, lr),
    "SGD": lambda model_params, lr: optim.SGD(model_params, lr),
    "RMSprop": lambda model_params, lr: optim.RMSprop(model_params, lr),
}
