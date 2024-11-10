import nbformat as nbf

# Load params from test.py
import sys
sys.path.append('.')  # Allow import from current directory
from test import params  # Import params from test.py
from params import DATALOADERS, LAYERS, ACTIVATIONS, LOSSES, OPTIMIZERS

# Initialize a new notebook
nb = nbf.v4.new_notebook()

layer_list = []
def parse_layer(layers):
    raw_layers = layers

    for l in raw_layers:
        component = None

        layer_type = l["kind"]
        if layer_type in LAYERS.keys():  # is a layer
            layer_args = l["args"]
            if layer_type == "Linear":
                i, o = layer_args
                component = LAYERS[layer_type](i, o)

            elif layer_type in ["Conv1D", "Conv2D", "Conv3D"]:
                i, o, k_size = layer_args
                component = LAYERS[layer_type](i, o, k_size)

            elif layer_type in ["LSTM", "GRU", "RNN"]:
                i, h_size = layer_args
                component = LAYERS[layer_type](i, h_size)

        elif layer_type in ACTIVATIONS.keys():  # is activation function
            component = ACTIVATIONS[layer_type]

        else:
            print("Invalid layer type")
            break

        layer_list.append(component)

# Install dependencies

install_cell = """
# Install necessary dependencies
%pip install torch numpy
"""

# Code cell 1: Import necessary libraries
code_cell_1 = f"""\
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from params import DATALOADERS, LAYERS, ACTIVATIONS, LOSSES, OPTIMIZERS
"""

# print(params["layers"])
# parse_layer(params["layers"])
# print(layer_list)

layers_code = ""
for layer in params["layers"]:
    if layer["kind"] == "Linear":
        layers_code += f'nn.Linear({layer["args"][0]}, {layer["args"][1]}),'
    elif layer["kind"] == "ReLU":
        layers_code += 'nn.ReLU(),'
    elif layer["kind"] == "Sigmoid":
        layers_code += 'nn.Sigmoid(),'

# Code cell 2:
code_cell_2 = f"""\
class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers_list = [
            {layers_code}
        ]
        self.layers = nn.ModuleList(layers_list)
    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x
"""


# Code cell 3:

code_cell_3 = f"""\
ds = DATALOADERS["{params["input"]}"]

X = torch.tensor(ds["X"], dtype=torch.float32)
Y = torch.tensor(ds["y"], dtype=torch.float32).reshape(-1, 1)

model = DynamicModel()
loss_fn = LOSSES["{params["loss"]}"]
optimizer = OPTIMIZERS["{params["optimizer"]["kind"]}"](
    model.parameters(), {params["optimizer"]["lr"]}
)

final_loss = -1
"""

# Code cell 4: Training loop
code_cell_4 = f"""\
# Training the model

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        batchX = X[i : i + batch_size]
        y_pred = model(batchX)
        y_batch = Y[i : i + batch_size]
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss = loss

print(f"Final loss: {{final_loss}}")
"""

# Add the generated code cells to the notebook
nb.cells.append(nbf.v4.new_code_cell(install_cell))
nb.cells.append(nbf.v4.new_code_cell(code_cell_1))
nb.cells.append(nbf.v4.new_code_cell(code_cell_2))
nb.cells.append(nbf.v4.new_code_cell(code_cell_3))
nb.cells.append(nbf.v4.new_code_cell(code_cell_4))

# Write the notebook to a file
with open("generated_notebook.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook generated as 'generated_notebook.ipynb'")
