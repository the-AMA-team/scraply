from flask import Flask, request, send_file
from models import (
    DynamicModel,
    Train,
    TransformerModel,
    TransformerData,
    TransformerTrain,
    Inference,
)
from flask_cors import CORS
from generate import Generate

# dumb imports that i gyatt to add
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


app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return {"data": "hello"}


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    try:
        gen = Generate(data)
        gen.generate_notebook()
        return send_file("generated_notebook.ipynb")
    except Exception as e:
        return {"status": "failed", "error": str(e)}


@app.post("/train")
def train():
    # example data
# data = {
#     "input": "MNIST",
#     "layers": [
#         {"kind": "Conv2D", "args": (1, 16, 3)},
#         {"kind": "ReLU"},
#         {"kind": "MaxPool2D", "args": (2, 2)},
#         {"kind": "Conv2D", "args": (16, 32, 3)},
#         {"kind": "ReLU"},
#         {"kind": "MaxPool2D", "args": (2, 2)},
#         {"kind": "Flatten", "args": [1,-1]},
#         {"kind": "Linear", "args": (800, 128)}, # supposed to be 32 * 7 * 7
#         {"kind": "ReLU"},
#         {"kind": "Linear", "args": (128, 10)}, 
#     ],
#     "loss": "CrossEntropy",
#     "optimizer": {"kind": "Adam", "lr": 0.001},
#     "epoch": 5,
#     "batch_size": 64,
# }

    data = request.get_json()
    print("Received data:", data)

    inp = data["input"]
    layers = data["layers"]
    loss = data["loss"]
    optimizer = data["optimizer"]
    n_epochs = data["epoch"]
    batch_size = data["batch_size"]

    RESULTS = {}

    try:
        model = DynamicModel(layers)

        t = Train(
            model=model,
            input=inp,
            loss=loss,
            optimizer=optimizer,
            batch_size=batch_size,
        )

        print("slay")
        RESULTS = t.train_test_log(n_epochs, batch_size)

    except Exception as e:
        print("Error:", e)
        RESULTS = {"error": str(e)}

    return {
        "RESULTS": RESULTS,
    }  # training loss


@app.post("/transformertrain")  # MODEL IS MOVED TO DEVICE INSIDE OF TRAIN FUNCTION
def transformertrain():
    # # example arguments
    # embed_dim = 100
    # heads = 2
    # hidden_dim = 2048
    # # example data
    # params = {
    #     "input": "alice", # preprocess
    #     "layers": [
    #         {"kind": "Decoder", "args": (embed_dim, heads, hidden_dim)},
    #         {"kind": "Decoder", "args": (embed_dim, heads, hidden_dim)},
    #         {"kind": "Output", "args": 0.3},
    #     ],
    #     "loss": "CrossEntropy",
    #     "optimizer": {"kind": "Adam", "lr": 0.001},
    #     "epoch": 100,
    #     "batch_size": 32,
    # }

    data = request.get_json()
    print("Received data:", data)

    inp = data["input"]
    layers = data["layers"]
    loss = data["loss"]
    optimizer = data["optimizer"]
    n_epochs = data["epoch"]
    batch_size = data["batch_size"]

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # clear GPU memory

        dataset = TransformerData(inp)
        model = TransformerModel(
            layers, dataset.vocab_size, dataset.sequence_length
        )  # model is moved to device in train function

        t = TransformerTrain(
            model=model,
            inp=inp,
            loss=loss,
            optimizer=optimizer,
            batch_size=batch_size,
        )

        print("it worked!")

        RESULTS = t.train(n_epochs)

    except Exception as e:
        print("Error:", e)
        RESULTS = {"error": str(e)}

    return {
        "RESULTS": RESULTS,
    }


@app.post("/transformertest")  # MODEL IS MOVED TO DEVICE INSIDE OF INFERENCE FUNCTION
def transformertest():
    infer_data = request.get_json()
    print("Received data:", infer_data)

    # PRE DEFINED MODEL FOR DEMO PURPOSES
    # example arguments
    embed_dim = 100
    heads = 2
    hidden_dim = 2048
    # example data
    data = {
        "input": "alice",  # preprocess
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

    inp = data["input"]
    layers = data["layers"]

    temperature = infer_data["temperature"]
    prompt = infer_data["prompt"]
    generate_length = 100  # this should be an actual argument in the future
    RESULTS = {}

    # hardcode example decoder model for now

    try:
        dataset = TransformerData(data["input"])
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # clear GPU memory

        model = TransformerModel(
            data["layers"], dataset.vocab_size, dataset.sequence_length
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.load_state_dict(torch.load("datasets/model2.pth", weights_only=True, map_location=device))
        

        print("Model loaded successfully!")

        word_to_int = dataset.word_to_int
        int_to_word = dataset.int_to_word
        SEQUENCE_LENGTH = dataset.sequence_length
        
        model.to(device) # move model to device

        text_gen = Inference(model, word_to_int, int_to_word, SEQUENCE_LENGTH)
        sample = text_gen.generate_text(
            prompt, generate_length, temperature=temperature, top_k=None
        )

        RESULTS = {"text": sample}

    except Exception as e:
        print("Error:", e)
        RESULTS = {"error": str(e)}

    return RESULTS