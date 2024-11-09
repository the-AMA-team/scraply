import torch.nn as nn
import torch.optim as optim
import pandas as pd


def are_params_valid(data):
    if data["input"] not in DATASETS.keys():
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


DATASETS = {
    "pima": {
        "X": pd.read_csv("datasets/pima-indians-diabetes.csv").iloc[:, :-1].values,
        "y": pd.read_csv("datasets/pima-indians-diabetes.csv").iloc[:, -1].values
    }
}


ACTIVATIONS = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Softmax": nn.Softmax(),
    "LeakyReLU": nn.LeakyReLU()
}


LAYERS = {
    "Linear": lambda i, o: nn.Linear(i, o),
    "Conv2D": lambda i, o, k_size: nn.Conv2d(i, o, k_size),
    "Conv1D": lambda i, o, k_size: nn.Conv1d(i, o, k_size),
    "Conv3D": lambda i, o, k_size: nn.Conv3d(i, o, k_size),
    "LSTM": lambda i, h_size: nn.LSTM(i, h_size),
    "GRU": lambda i, h_size: nn.GRU(i, h_size),
    "RNN": lambda i, h_size: nn.RNN(i, h_size)
}


LOSSES = {
    "BCE": nn.BCELoss()
}


OPTIMIZERS = {
    "Adam": lambda model_params, lr: optim.Adam(model_params, lr)
}
