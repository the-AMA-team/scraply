import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from params import DATASETS, LAYERS, ACTIVATIONS, LOSSES, OPTIMIZERS

# expected data example from the api
data = {
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
    "optimizer": "Adam",
    "epoch": 100,
    "batch_size": 10, 
}

class DynamicModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        raw_layers = layers
        self.layer_list = []

        for l in raw_layers:
            component = None

            layer_type = l["kind"]
            if layer_type in LAYERS.keys(): # is a layer 
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

            elif layer_type in ACTIVATIONS.keys(): # is activation function
                component = ACTIVATIONS[layer_type] 

            else: 
                print("Invalid layer type")
                break

            self.layer_list.append(component)

        self.layers = nn.ModuleList(self.layer_list)

    
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        
        return x


# this would be imported into app.py where a single function will be called to give in the 
# data object, returned would be the final loss for now. 
# would the model need to be stored temporarily so that it can be accessed after training to make predictions? how can models be stored?
# if we build the prediction to be a class instance is it a good idea to just store it as a bin file???

# TODO: make this dynamic wrt sample data 

class Train:
    def __init__(self, model, input, loss, optimizer):
        # validify data before this point, each valid key should exist and errors for invalid
        ds = DATASETS[input]

        self.X = torch.tensor(ds["X"], dtype=torch.float32)
        self.Y = torch.tensor(ds["y"], dtype=torch.float32).reshape(-1, 1)

        self.model = model
        self.loss_fn = LOSSES[loss]
        self.optimizer = OPTIMIZERS[optimizer["kind"]](self.model.parameters(), optimizer["lr"])

        self.final_loss = -1

    def train(self, n_epochs, batch_size):
        for epoch in range(n_epochs):
            for i in range(0, len(self.X), batch_size):
                batchX = self.X[i:i+batch_size]
                y_pred = self.model(batchX)
                y_batch = self.Y[i:i+batch_size]
                loss = self.loss_fn(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.final_loss = loss

        return self.final_loss


if __name__ == "__main__":
    model = DynamicModel(data["layers"])
    print(model)
    t = Train(model=model, input=data["input"], loss=data["loss"], optimizer=data["optimizer"])
    fl = t.train(n_epochs=100, batch_size=10)
    print(f'Final loss: {fl}')
