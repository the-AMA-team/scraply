import nbformat as nbf

# Load params from test.py
import sys
sys.path.append('.')  # Allow import from current directory
# from test import params  # Import params from test.py
from params import DATALOADERS, LAYERS, ACTIVATIONS, LOSSES, OPTIMIZERS

# format of tax
# params = {
#     "input": "pima",
#     "layers": [
#         {"kind": "Linear", "args": (8, 12)},
#         {"kind": "ReLU"},
#         {"kind": "Linear", "args": (12, 8)},
#         {"kind": "ReLU"},
#         {"kind": "Linear", "args": (8, 1)},
#         {"kind": "Sigmoid"},
#     ],
#     "loss": "BCE",
#     "optimizer": {"kind": "Adam", "lr": 0.001},
#     "epoch": 3,
#     "batch_size": 10,
# }

# Install dependencies

install_cell = """
# Install necessary dependencies
%pip install torch numpy scikit-learn
"""

# Code cell 1: Import necessary libraries and set test
code_cell_1 = f"""\
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
"""

add_params = f"""\

DATALOADERS = {{
    "pima": {{
        "X": pd.read_csv("datasets/pima-indians-diabetes.csv").iloc[:, :-1].values,
        "y": pd.read_csv("datasets/pima-indians-diabetes.csv").iloc[:, -1].values,
    }},
    "MNIST": {{
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
    }},
    "FashionMNIST": {{
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
    }},
    "CIFAR10": {{
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
    }},
}}


ACTIVATIONS = {{
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "Softmax": nn.Softmax(),
    "LeakyReLU": nn.LeakyReLU(),
    "PReLU": nn.PReLU(),
}}


LAYERS = {{
    "Flatten": nn.Flatten(), # no argumnets for Flatten()
    "Linear": lambda i, o: nn.Linear(i, o),
    "Conv2D": lambda i, o, k_size: nn.Conv2d(i, o, k_size), # i = input channels (1 --> grayscale, 3 --> RGB), o = output channels (number of filters), k_size = kernel size
    "Conv1D": lambda i, o, k_size: nn.Conv1d(i, o, k_size),
    "Conv3D": lambda i, o, k_size: nn.Conv3d(i, o, k_size),
    "LSTM": lambda i, h_size: nn.LSTM(i, h_size),
    "GRU": lambda i, h_size: nn.GRU(i, h_size),
    "RNN": lambda i, h_size: nn.RNN(i, h_size),
    "Dropout": lambda p: nn.Dropout(p),
}}


LOSSES = {{
    "BCE": nn.BCELoss(),  # binary cross entropy --> 0 or 1 classification models
    "CrossEntropy": nn.CrossEntropyLoss(),  # multi-class classification models
    # "MSE": nn.MSELoss() # regression models
}}


OPTIMIZERS = {{
    "Adam": lambda model_params, lr: optim.Adam(
        model_params, lr
    ),  # momentum parameter is optional
    "AdamW": lambda model_params, lr: optim.AdamW(model_params, lr),
    "SGD": lambda model_params, lr: optim.SGD(model_params, lr),
    "RMSprop": lambda model_params, lr: optim.RMSprop(model_params, lr),
}}
"""

# print(params["layers"])
# parse_layer(params["layers"])
# print(layer_list)


# Code cell 4: Training loop
code_cell_4 = f"""\
# Training the model

def train(self, n_epochs, batch_size):
    # if self.input == "pima":
    #     # PIMA TRAINING FUNCTION
    #     for epoch in range(n_epochs):
    #         for i in range(0, len(self.X), batch_size):
    #             batchX = self.X[i : i + batch_size]
    #             y_pred = self.model(batchX)
    #             y_batch = self.Y[i : i + batch_size]
    #             loss = self.loss_fn(y_pred, y_batch)
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()

    #             self.final_loss = loss

    #     return self.final_loss
    # else:
    #     # IMAGE DATASETS TRAINING FUNCTION
    size = len(self.train_loader.dataset)
    # num_batches = len(self.train_loader)
    self.model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch, (X, y) in enumerate(self.train_loader):
        X, y = X.to(self.device), y.to(self.device)
        # Compute prediction error
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        train_loss += loss.item()
        # Calculate accuracy
        _, predicted = torch.max(
            pred, 1
        )  # Get the predicted class (index with max value)
        correct += (predicted == y).sum().item()  # Count correct predictions
        total += y.size(0)  # Count total predictions

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {{loss:>7f}}  [{{current:>5d}}/{{size:>5d}}]")

    # Average loss over all batches
    avg_train_loss = train_loss / len(self.train_loader)
    # Calculate accuracy as a percentage
    avg_acc = 100 * correct / total
    return avg_train_loss, avg_acc

Train.train = train
"""

code_cell_5 = f"""\
def test(self, n_epochs, batch_size):
    # if self.input == "pima":
    #     # PIMA TRAINING FUNCTION
    #     # do nothing, because pima currently does not have a test_dataset (only has X, y dataset --> has not been split yet)
    #     print("PIMA is not configured to have a test dataset yet")
    # else:
    size = len(self.test_loader.dataset)
    num_batches = len(self.test_loader)
    self.model.eval() # model mode change is especially important for dropout layer 
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in self.test_loader:
            X, y = X.to(self.device), y.to(self.device)
            # Compute prediction error
            pred = self.model(X)
            test_loss += self.loss_fn(pred, y).item()
            correct += (
                (pred.argmax(1) == y).type(torch.float).sum().item()
            )  # for accuracy

    test_loss /= num_batches
    correct /= size
    avg_acc = 100 * correct

    # Print loss & accuracy
    print(
        f"Test Error: \\n Accuracy: {{(avg_acc):>0.1f}}%, Avg loss: {{test_loss:>8f}} \\n"
    )

    # Average loss over all batches
    avg_test_loss = test_loss / len(self.test_loader)
    return avg_test_loss, avg_acc
Train.test = test
"""

code_cell_6 = f"""\
def train_test_log(self, n_epochs, batch_size):
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        for t in range(n_epochs):
            print(f"Epoch {{t+1}}\\n-------------------------------")
            avg_train_loss, train_avg_acc = self.train(n_epochs, batch_size)
            avg_test_loss, test_avg_acc = self.test(n_epochs, batch_size)

            # Store losses
            train_losses.append(avg_train_loss)
            train_accs.append(train_avg_acc)
            test_losses.append(avg_test_loss)
            test_accs.append(test_avg_acc)

        # calculate average accuracy and average loss
        avg_train_acc = sum(train_accs) / len(train_accs)
        avg_test_acc = sum(test_accs) / len(test_accs)
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_test_loss = sum(test_losses) / len(test_losses)

        print("Done!")

        return {{
            "train_losses": train_losses,
            "test_losses": test_losses,
            "avg_train_loss": avg_train_loss,
            "avg_test_loss": avg_test_loss,
            "avg_train_acc": avg_train_acc,
            "avg_test_acc": avg_test_acc,
        }}

Train.train_test_log = train_test_log
        # can add more information to this dictionary, like the saved model, best epochs, etc.
"""



code_cell_8 = f"""\
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs + 1), train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()
"""

class Generate:

    def __init__(self, params):
        self.params = params
        self.layers_code = ""
        for layer in params["layers"]:
            if layer["kind"] == "Linear":
                self.layers_code += f'nn.Linear({layer["args"][0]}, {layer["args"][1]}),'
            elif layer["kind"] == "ReLU":
                self.layers_code += 'nn.ReLU(),'
            elif layer["kind"] == "Sigmoid":
                self.layers_code += 'nn.Sigmoid(),'

        
        
        # Initialize a new notebook

    def generate_notebook(self):
        # Add the generated code cells to the notebook

        # Code cell 2:
        code_cell_2 = f"""\
class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers_list = [
            {self.layers_code}
        ]
        self.layers = nn.ModuleList(layers_list)
    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x
"""

        # Code cell 3: instantiate training class

        code_cell_3 = f"""\
class Train:
    def __init__(self, model, input, loss, optimizer, batch_size):
        # validate data inputs before this point, each valid key should exist and errors for invalid
        self.input = input
        ds = DATALOADERS["{self.params["input"]}"]

        self.device = (  # for GPU access --> works with CPU as well
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {{self.device}} device")

        # MOVE MODEL TO DEVICE
        self.model = model.to(self.device)

        # preprocessing data here!!!
        if input == "pima":
            X = ds["X"]
            y = ds["y"]

            # split test and training data using ski-kit learn module
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            # could normalize the data here
            # create tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(
                -1, 1
            )  # Reshape for binary classification
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
            # create dataset objects
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            # create dataLoader objects
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            train_set = ds[
                "train"
            ]  # ds["train"] is a dataset object, already transformed into tensor
            test_set = ds["test"]
            self.train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True
            )
            self.test_loader = DataLoader(
                test_set, batch_size=batch_size, shuffle=False
            )

        self.loss_fn = LOSSES["{self.params["loss"]}"]
        self.optimizer = OPTIMIZERS["{self.params["optimizer"]["kind"]}"](
            self.model.parameters(), {self.params["optimizer"]["lr"]}
        )

        self.final_loss = -1
                    """
        
        code_cell_7 = f"""\
if __name__ == "__main__":

    model = DynamicModel()
    # model = DynamicModel(params["layers"]).to(device)
    print(model)
    t = Train(
        model=model,
        input="{self.params["input"]}",
        loss="{self.params["loss"]}",
        optimizer="{self.params["optimizer"]["kind"]}",
        batch_size={self.params["batch_size"]},
    )

    RESULTS = t.train_test_log({self.params["epoch"]}, batch_size={self.params["batch_size"]})
    print("Results:")
    print("Average Training Loss: ", RESULTS["avg_train_loss"])
    print("Average Testing Loss: ", RESULTS["avg_test_loss"])
    print("Average Training Accuracy: ", RESULTS["avg_train_acc"])
    print("Average Testing Accuracy: ", RESULTS["avg_test_acc"])
"""

        nb = nbf.v4.new_notebook()
        nb.cells.append(nbf.v4.new_code_cell(install_cell))
        nb.cells.append(nbf.v4.new_code_cell(code_cell_1))
        nb.cells.append(nbf.v4.new_code_cell(add_params))
        nb.cells.append(nbf.v4.new_code_cell(code_cell_2))
        nb.cells.append(nbf.v4.new_code_cell(code_cell_3))
        nb.cells.append(nbf.v4.new_code_cell(code_cell_4))
        nb.cells.append(nbf.v4.new_code_cell(code_cell_5))
        nb.cells.append(nbf.v4.new_code_cell(code_cell_6))
        nb.cells.append(nbf.v4.new_code_cell(code_cell_7))

        # Write the notebook to a file
        with open("generated_notebook.ipynb", "w") as f:
            nbf.write(nb, f)

        print("Notebook generated as 'generated_notebook.ipynb'")

if __name__ == '__main__':
    params = {
        "input": "pima",
        "layers": [
            {"kind": "Linear", "args": (8, 12)},
            {"kind": "ReLU"},
            {"kind": "Linear", "args": (12, 8)},
            {"kind": "ReLU"},
            {"kind": "Linear", "args": (8, 1)},
            {"kind": "Sigmoid"},
        ],
        "loss": "BCE",
        "optimizer": {"kind": "Adam", "lr": 0.001},
        "epoch": 3,
        "batch_size": 10,
    }

    gen = Generate(params)
    gen.generate_notebook()
