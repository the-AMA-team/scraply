import nbformat as nbf
import sys

sys.path.append(".")
from params import LAYERS, ACTIVATIONS, LOSSES, OPTIMIZERS


class Generate:
    def __init__(self, params):
        self.params = params
        self.layers_code = self._generate_layers_code()
        self.dataset_info = self._get_dataset_info()

    def _generate_layers_code(self):
        """Generate the layers code based on the architecture parameters"""
        layers_code = ""
        for layer in self.params["layers"]:
            if layer["kind"] == "Linear":
                layers_code += (
                    f"nn.Linear({layer['args'][0]}, {layer['args'][1]}),\n            "
                )
            elif layer["kind"] == "ReLU":
                layers_code += "nn.ReLU(),\n            "
            elif layer["kind"] == "Sigmoid":
                layers_code += "nn.Sigmoid(),\n            "
            elif layer["kind"] == "Tanh":
                layers_code += "nn.Tanh(),\n            "
            elif layer["kind"] == "Conv2D":
                # args: (dim, input_channels, output_channels, kernel_size, stride, padding)
                layers_code += f"nn.Conv2d({layer['args'][1]}, {layer['args'][2]}, {layer['args'][3]}, stride={layer['args'][4]}, padding={layer['args'][5]}),\n            "
            elif layer["kind"] == "MaxPool2D":
                # args: (dim, kernel_size, stride, padding)
                layers_code += f"nn.MaxPool2d({layer['args'][1]}, stride={layer['args'][2]}),\n            "
            elif layer["kind"] == "Flatten":
                layers_code += "nn.Flatten(),\n            "
            elif layer["kind"] == "Dropout":
                layers_code += f"nn.Dropout({layer['args'][0]}),\n            "
            elif layer["kind"] == "LeakyReLU":
                layers_code += "nn.LeakyReLU(),\n            "
            elif layer["kind"] == "PReLU":
                layers_code += "nn.PReLU(),\n            "

        return layers_code.rstrip(",\n            ") + "\n        "

    def _get_dataset_info(self):
        """Get dataset-specific information"""
        dataset_name = self.params["input"]
        if dataset_name == "pima":
            return {
                "type": "tabular",
                "input_features": 8,
                "num_classes": 2,
                "data_shape": "tabular",
                "description": "PIMA Indians Diabetes Dataset - Binary classification",
            }
        elif dataset_name in ["MNIST", "FashionMNIST"]:
            return {
                "type": "image",
                "input_features": 784,  # 28x28 flattened
                "num_classes": 10,
                "data_shape": "(1, 28, 28)",  # grayscale
                "description": f"{dataset_name} - 10-class image classification",
            }
        elif dataset_name == "CIFAR10":
            return {
                "type": "image",
                "input_features": 3072,  # 3x32x32 flattened
                "num_classes": 10,
                "data_shape": "(3, 32, 32)",  # RGB
                "description": "CIFAR-10 - 10-class color image classification",
            }
        else:
            return {
                "type": "unknown",
                "input_features": "unknown",
                "num_classes": "unknown",
                "data_shape": "unknown",
                "description": f"Unknown dataset: {dataset_name}",
            }

    def _generate_dataset_cell(self):
        """Generate dataset-specific initialization code"""
        dataset_name = self.params["input"]
        batch_size = self.params["batch_size"]

        if dataset_name == "pima":
            return f"""# Dataset Setup - PIMA Indians Diabetes (Tabular)
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Download and load PIMA dataset from URL
url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
data = pd.read_csv(url)
X = data.iloc[:, :-1].values  # All columns except last
y = data.iloc[:, -1].values   # Last column (target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
import torch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Create datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)

print(f"PIMA Dataset loaded from URL:")
print(f"  Training samples: {{len(train_dataset)}}")
print(f"  Test samples: {{len(test_dataset)}}")
print(f"  Input features: {{X_train.shape[1]}}")
print(f"  Classes: Binary (0/1)")"""

        else:  # Image datasets
            return f"""# Dataset Setup - {dataset_name} (Image)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load dataset
train_dataset = datasets.{dataset_name}(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.{dataset_name}(
    root="data", 
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size={batch_size}, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size={batch_size}, shuffle=False)

# Get sample data to understand dimensions
sample_data, _ = next(iter(train_loader))
print(f"{dataset_name} Dataset loaded:")
print(f"  Training samples: {{len(train_dataset)}}")
print(f"  Test samples: {{len(test_dataset)}}")
print(f"  Input shape: {{sample_data.shape[1:]}}")
print(f"  Number of classes: {{len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 'Unknown'}}")"""

    def _generate_model_cell(self):
        """Generate model architecture code"""
        dataset_info = self.dataset_info
        dataset_name = self.params["input"]

        if dataset_name == "pima":
            return f"""# Model Architecture for PIMA (Tabular)
import torch.nn as nn

class PimaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            {self.layers_code}
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = PimaModel()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Architecture:")
print(model)
print(f"\\nTotal parameters: {{total_params:,}}")
print(f"Trainable parameters: {{trainable_params:,}}")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"Model moved to: {{device}}")"""
        else:
            return f"""# Model Architecture for {dataset_name} (Image)
import torch.nn as nn

class {dataset_name.capitalize()}Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            {self.layers_code}
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = {dataset_name.capitalize()}Model()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model Architecture:")
print(model)
print(f"\\nTotal parameters: {{total_params:,}}")
print(f"Trainable parameters: {{trainable_params:,}}")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"Model moved to: {{device}}")"""

    def _generate_training_cell(self):
        """Generate training configuration and functions"""
        dataset_name = self.params["input"]
        if dataset_name == "pima":
            return f"""# Training Configuration for PIMA (Tabular)
import torch.optim as optim
import torch.nn as nn

epochs = {self.params["epoch"]}
learning_rate = {self.params["optimizer"]["lr"]}
batch_size = {self.params["batch_size"]}

loss_function = nn.BCELoss()
optimizer = optim.{self.params["optimizer"]["kind"]}(model.parameters(), lr=learning_rate)

print("Training Configuration:")
print(f"  Epochs: {{epochs}}")
print(f"  Learning Rate: {{learning_rate}}")
print(f"  Loss Function: BCE")
print(f"  Optimizer: {self.params["optimizer"]["kind"]}")
print(f"  Batch Size: {{batch_size}}")

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # For BCE, output is sigmoid, so threshold at 0.5
        preds = (output > 0.5).float()
        correct += (preds == target).sum().item()
        total += target.size(0)
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            preds = (output > 0.5).float()
            correct += (preds == target).sum().item()
            total += target.size(0)
    return total_loss / len(test_loader), 100. * correct / total"""
        else:
            return f"""# Training Configuration for {dataset_name} (Image)
import torch.optim as optim
import torch.nn as nn

epochs = {self.params["epoch"]}
learning_rate = {self.params["optimizer"]["lr"]}
batch_size = {self.params["batch_size"]}

loss_function = nn.CrossEntropyLoss()
optimizer = optim.{self.params["optimizer"]["kind"]}(model.parameters(), lr=learning_rate)

print("Training Configuration:")
print(f"  Epochs: {{epochs}}")
print(f"  Learning Rate: {{learning_rate}}")
print(f"  Loss Function: CrossEntropy")
print(f"  Optimizer: {self.params["optimizer"]["kind"]}")
print(f"  Batch Size: {{batch_size}}")

def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = output.argmax(dim=1, keepdim=True)
        correct += preds.eq(target.view_as(preds)).sum().item()
        total += target.size(0)
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()
            total += target.size(0)
    return total_loss / len(test_loader), 100. * correct / total"""

    def _generate_training_loop_cell(self):
        """Generate the main training loop"""
        dataset_name = self.params["input"]
        if dataset_name == "pima":
            return f"""# Training Loop for PIMA (Tabular)
import matplotlib.pyplot as plt

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print("Starting training...")
print("=" * 50)

for epoch in range(1, {self.params["epoch"]} + 1):
    train_loss, train_acc = train_epoch(model, train_loader, loss_function, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, loss_function, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    print(f'Epoch {{epoch:2d}}/{self.params["epoch"]}:')
    print(f'  Train Loss: {{train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%')
    print(f'  Test Loss: {{test_loss:.4f}}, Test Acc: {{test_acc:.2f}}%')
    print("-" * 50)

print("Training completed!")

final_train_acc = train_accuracies[-1]
final_test_acc = test_accuracies[-1]
overfitting = final_train_acc - final_test_acc

print("\\nFinal Results:")
print(f"  Final Training Accuracy: {{final_train_acc:.2f}}%")
print(f"  Final Test Accuracy: {{final_test_acc:.2f}}%")
print(f"  Overfitting gap: {{overfitting:.2f}}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(range(1, {self.params["epoch"]} + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
ax1.plot(range(1, {self.params["epoch"]} + 1), test_losses, 'r-', label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(range(1, {self.params["epoch"]} + 1), train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
ax2.plot(range(1, {self.params["epoch"]} + 1), test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
torch.save(model.state_dict(), 'trained_model.pth')
print("\\nModel saved as 'trained_model.pth'")"""
        else:
            return f"""# Training Loop for {dataset_name} (Image)
import matplotlib.pyplot as plt

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print("Starting training...")
print("=" * 50)

for epoch in range(1, {self.params["epoch"]} + 1):
    train_loss, train_acc = train_epoch(model, train_loader, loss_function, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, loss_function, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    print(f'Epoch {{epoch:2d}}/{self.params["epoch"]}:')
    print(f'  Train Loss: {{train_loss:.4f}}, Train Acc: {{train_acc:.2f}}%')
    print(f'  Test Loss: {{test_loss:.4f}}, Test Acc: {{test_acc:.2f}}%')
    print("-" * 50)

print("Training completed!")

final_train_acc = train_accuracies[-1]
final_test_acc = test_accuracies[-1]
overfitting = final_train_acc - final_test_acc

print("\\nFinal Results:")
print(f"  Final Training Accuracy: {{final_train_acc:.2f}}%")
print(f"  Final Test Accuracy: {{final_test_acc:.2f}}%")
print(f"  Overfitting gap: {{overfitting:.2f}}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(range(1, {self.params["epoch"]} + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
ax1.plot(range(1, {self.params["epoch"]} + 1), test_losses, 'r-', label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax2.plot(range(1, {self.params["epoch"]} + 1), train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
ax2.plot(range(1, {self.params["epoch"]} + 1), test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
torch.save(model.state_dict(), 'trained_model.pth')
print("\\nModel saved as 'trained_model.pth'")"""

    def generate_notebook(self):
        """Return notebook JSON for the user's architecture (not written to disk)."""

        # Cell 1: Title and Overview
        title_cell = nbf.v4.new_markdown_cell(f"""# {self.params["input"].upper()} Neural Network Training

## Architecture Overview
This notebook implements a neural network for **{self.dataset_info["description"]}**

### Configuration:
- **Dataset**: {self.params["input"]}
- **Architecture**: {len(self.params["layers"])} layers
- **Loss Function**: {self.params["loss"]}
- **Optimizer**: {self.params["optimizer"]["kind"]} (lr={self.params["optimizer"]["lr"]})
- **Training**: {self.params["epoch"]} epochs, batch size {self.params["batch_size"]}

### Layer Structure:
""")

        # Add layer details to markdown
        for i, layer in enumerate(self.params["layers"]):
            title_cell.source += f"- Layer {i + 1}: {layer['kind']}"
            if "args" in layer:
                title_cell.source += f" with args {layer['args']}"
            title_cell.source += "\n"

        # Cell 2: Install Dependencies
        install_cell = nbf.v4.new_code_cell("""# Install required packages
%pip install torch torchvision numpy scikit-learn matplotlib pandas""")

        # Cell 3: Import Libraries
        imports_cell = nbf.v4.new_code_cell("""# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")""")

        # Cell 4: Dataset Setup (dynamically generated)
        dataset_cell = nbf.v4.new_code_cell(self._generate_dataset_cell())

        # Cell 5: Model Architecture (dynamically generated)
        model_cell = nbf.v4.new_code_cell(self._generate_model_cell())

        # Cell 6: Training Configuration (dynamically generated)
        training_cell = nbf.v4.new_code_cell(self._generate_training_cell())

        # Cell 7: Training Loop (dynamically generated)
        training_loop_cell = nbf.v4.new_code_cell(self._generate_training_loop_cell())

        nb = nbf.v4.new_notebook()
        nb.cells = [
            title_cell,
            install_cell,
            imports_cell,
            dataset_cell,
            model_cell,
            training_cell,
            training_loop_cell,
        ]
        return nbf.writes(nb)
