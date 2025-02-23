import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split  # --> pip install scikit-learn
import math
from collections import Counter

from params import DATALOADERS, LAYERS, ACTIVATIONS, LOSSES, OPTIMIZERS

# data loader + suggestions
# expected data example from the api


class DynamicModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        raw_layers = layers
        self.layer_list = []

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

                elif layer_type == "Dropout":
                    component = LAYERS[layer_type](p=layer_args)  # 1 arg

                elif layer_type == "Flatten":
                    component = LAYERS[layer_type]  # no args needed

            elif layer_type in ACTIVATIONS.keys():  # is activation function
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


class TransformerModel(nn.Module):
    # embed_dim, heads, hidden_dim
    # get vocab_size & SEQUENCE_LENGTH from data procressing

# sequential = nn.Sequential(*modules)

    def __init__(self, userlayers, vocab_size, SEQUENCE_LENGTH):
        super(TransformerModel, self).__init__()
        
        # need to parse through user layers first to access embed_dim for other layers
        # ----- user defined decoders ------
        for l in userlayers:
            layer_type = l["kind"]
            if layer_type in LAYERS.keys():  # is a layer
                layer_args = l["args"]
                if layer_type == "Decoder":
                    embed_dim, heads, hidden_dim = layer_args
                    self.embed_dim = embed_dim # um this updates everytime because im lazy
                    decoder_layer = LAYERS[layer_type](embed_dim, heads, hidden_dim, batch_first=True) # not sure if batch_first = True is neccessary
                    self.decoder_layers.append(decoder_layer)
                else: # ------------  output layer ---------- ---> assuming that decoders/encoders come first and output layers come last
                    # dropout
                    # linear layer
                    if layer_type == "Output":
                        p = layer_args
                        self.dropout_layer = LAYERS[layer_type](p)
                        self.linear_layer = nn.Linear(embed_dim, vocab_size) # logits of the next word prediction

        self.pos_encoder = PositionalEncoding(max_len=SEQUENCE_LENGTH, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim) # OUTPUT: [batch_size, sequence_length, 100]
        # torch.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=<function relu>, layer_norm_eps=1e-05, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
        
        self.decoder_layers = nn.ModuleList()
        

    def forward(self, x):
        emb = self.emb(x) # embedding
        input_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device) # make input mask
        x = self.pos_encoder(emb)
        # decoder initialization time!
        # x = self.decoder_layer(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        for decoder in self.decoder_layers:
            x = decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        
        x = self.dropout_layer(x)
        out = self.linear_layer(x)
        
        return out

    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    

# if type = 'transformer' then run using this function:
# class TransformerModel(nn.Module):
#     def __init__(self, layers):
#         super().__init__()

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        """
        :param max_len: Input length sequence.
        :param d_model: Embedding dimension.
        :param dropout: Dropout value (default=0.1)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # first generate positional encodings
        return self.dropout(x) # do some dropout i guess
        #     input: [sequence length, batch size, embed dim]
        #     output: [sequence length, batch size, embed dim]

# run if layer = decoder
# class TransformerModel

# class TransformerTrain
# class Transformer
class TransformerData(Dataset): 
    def __init__(self, inp):
        self.vocab_size, self.sequence_length, self.word_to_int, self.int_to_word, self.samples = self.txt_dataset(inp)
        
    def __len__(self):
        return len(self.samples) # number of samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx] # retrieving ith sample
        input_seq = torch.LongTensor([self.word_to_int[word] for word in sample[:-1]]) # input
        target_seq = torch.LongTensor([self.word_to_int[word] for word in sample[1:]]) # target words (slides over by 1 each time)
        # remember --> only one target is being outputted each time!
        return input_seq, target_seq
    
    def txt_dataset(inp): 
        if(inp == "alice"):
            file_path = "datasets/alice_1.txt"
        if(inp == "shakespeare"):
            file_path = "datasets/shakespeare.txt"
            
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        # tokenize the text into words
        words = text.split() 
        # count unique words from text
        word_counts = Counter(words)
        # make list of the unique words ---> to create a vocabulary
        vocab = list(word_counts.keys())
        VOCAB_SIZE = len(vocab)
        SEQUENCE_LENGTH = 64
        WORD_TO_INT = {word: i for i, word in enumerate(vocab)} # maps each word to a unique integer index
        INT_TO_WORD = {i: word for word, i in WORD_TO_INT.items()} # maps each integer to a word
        SAMPLES = [words[i:i+SEQUENCE_LENGTH+1] for i in range(len(words)-SEQUENCE_LENGTH)] # training samples of 64 word length
        
        return VOCAB_SIZE, SEQUENCE_LENGTH, WORD_TO_INT, INT_TO_WORD, SAMPLES
    



class TransformerTrain: # input is DATALOADERS 
    def __init__(self, model, inp, loss, optimizer, batch_size):
        self.dataset = TransformerData(inp)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = batch_size,
            shuffle = True,
        )

        self.device = (  # for GPU access --> works with CPU as well
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")

        # MOVE MODEL TO DEVICE
        self.model = model.to(self.device)

        self.loss_fn = LOSSES[loss]
        self.optimizer = OPTIMIZERS[optimizer["kind"]](self.model.parameters(), optimizer["lr"])
        
        #print(model)
    
    def train(self, n_epochs):
        size = len(self.dataloader.dataset)
        
        self.model.train()
        
        train_loss = []
        
        for epoch in range(n_epochs):
            running_loss = 0
            for input_seq, target_seq in self.dataloader:
                input_seq, target_seq = input_seq.to(self.device), target_seq.to(self.device)
                outputs = model(input_seq)
                target_seq = target_seq.contiguous().view(-1)
                outputs = outputs.view(-1, self.dataset.vocab_size)

                loss = self.loss_fn(outputs, target_seq.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.detach().cpu().numpy()
            epoch_loss = running_loss / len(self.dataloader)
            print(f"Epoch {epoch} loss: {epoch_loss:.3f}")
            train_loss.append(epoch_loss)
            
        print("Done!")
        # torch.cuda.empty_cache()
        
        return {"train_loss": train_loss}
    
    
class Train:
    def __init__(self, model, input, loss, optimizer, batch_size):
        self.input = input
        ds = DATALOADERS[input]

        self.device = (  # for GPU access --> works with CPU as well
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        print(f"Using {self.device} device")

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
            train_set = ds["train"]
            test_set = ds["test"]
            self.train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True
            )
            self.test_loader = DataLoader(
                test_set, batch_size=batch_size, shuffle=False
            )

        self.loss_fn = LOSSES[loss]
        self.optimizer = OPTIMIZERS[optimizer["kind"]](
            self.model.parameters(), optimizer["lr"]
        )

        self.final_loss = -1

    def train(self, n_epochs, batch_size):
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

            if self.input == "pima":
                predicted = (
                    pred > 0.5
                ).float()  # apply threshold for binary classification
            else:
                _, predicted = torch.max(pred, 1)  # for multi-class classification

            # Get the predicted class (index with max value)
            correct += (predicted == y).sum().item()  # Count correct predictions
            total += y.size(0)  # Count total predictions

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # Average loss over all batches
        avg_train_loss = train_loss / len(self.train_loader)
        # Calculate accuracy as a percentage
        avg_acc = 100 * correct / total
        return avg_train_loss, avg_acc

    def test(self, n_epochs, batch_size):
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        self.model.eval()  # model mode change is especially important for dropout layers
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                # Compute prediction error
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                if self.input == "pima":
                    predicted = (pred > 0.5).type(torch.float)
                    correct += (predicted == y).sum().item()
                else:
                    correct += (
                        (pred.argmax(1) == y).type(torch.float).sum().item()
                    )  # for accuracy

        test_loss /= num_batches
        correct /= size
        avg_acc = 100 * correct

        # Print loss & accuracy
        print(
            f"Test Error: \n Accuracy: {(avg_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

        # Average loss over all batches
        avg_test_loss = test_loss / len(self.test_loader)
        return avg_test_loss, avg_acc

    def train_test_log(self, n_epochs, batch_size):
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []
        for t in range(n_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
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
        # torch.cuda.empty_cache()

        return {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "avg_train_loss": avg_train_loss,
            "avg_test_loss": avg_test_loss,
            "avg_train_acc": avg_train_acc,
            "avg_test_acc": avg_test_acc,
        }

        # can add more information to this dictionary, like the saved model, best epochs, etc.


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_loss_graph(train_losses, n_epochs):
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, n_epochs + 1), train_losses, label="Training Loss", color="blue"
        )
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    # example arguments
    embed_dim = 100
    heads = 2
    hidden_dim = 2048
    # example data
    params = {
        "type": "transformer", # ADDED NEW PARAMETER
        "input": "alice", # preprocess
        "layers": [
            {"kind": "Decoder", "args": (embed_dim, heads, hidden_dim)},
            {"kind": "Decoder", "args": (embed_dim, heads, hidden_dim)},
            {"kind": "Output", "args": 0.3},
        ],
        "loss": "CrossEntropy",
        "optimizer": {"kind": "Adam", "lr": 0.001},
        "epoch": 100,
        "batch_size": 32,
    }
    
    dataset = TransformerData(params["input"])

    model = TransformerModel(params["layers"], dataset.vocab_size, dataset.sequence_length) # model is moved to device in train function

    print(model)
    
    t = TransformerTrain(
        model=model,
        input=params["input"],
        loss=params["loss"],
        optimizer=params["optimizer"],
        batch_size=params["batch_size"],
    )

    RESULTS = t.train(params["epoch"])

    print("Results:")
    print("Training Loss: ", RESULTS["train_loss"])

    # In your main function or after calling train_test_log
    plot_loss_graph(RESULTS["train_loss"], params["epoch"])
