from flask import Flask, request, send_file
from models import (
    DynamicModel,
    Train,
    TransformerModel,
    TransformerData,
    TransformerTrain,
)
from flask_cors import CORS
from generate import Generate


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
    #     "epoch": 100,
    #     "batch_size": 10,
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
    } # training loss


@app.post("/transformertrain")
def transformertrain():
    # # example arguments
    # embed_dim = 100
    # heads = 2
    # hidden_dim = 2048
    # # example data
    # params = {
    #     "type": "transformer", # ADDED NEW PARAMETER
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
        dataset = TransformerData(inp)
        model = TransformerModel(
            layers, dataset.vocab_size, dataset.sequence_length
        )  # model is moved to device in train function

        t = TransformerTrain(
            model = model,
            inp = inp,
            loss = loss,
            optimizer = optimizer,
            batch_size = batch_size,
        )

        print("it worked!")

        RESULTS = t.train(n_epochs)

    except Exception as e:
        print("Error:", e)
        RESULTS = {"error": str(e)}

    return {
        "RESULTS": RESULTS,
    }




@app.post("/transformertest")
def transformertest():
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

    temperature = data["temperature"]
    prompt = data["prompt"]
    
    
    
    # hardcode example decoder model for now

    try:
        
        # initialize model here (using user architecture)
        # move model to device
        # model.load using state_dict
        
        # inference time!
        # use vocab_size, sequence_length, and int_to_word from data --> from request data
        # use temperature and prompt
        # generate text using model.generate_text()
        # return generated text
        
        
        
        
        
        
        dataset = TransformerData(inp)
        model = TransformerModel(
            layers, dataset.vocab_size, dataset.sequence_length
        )  # model is moved to device in train function

        t = TransformerTrain(
            model = model,
            inp = inp,
            loss = loss,
            optimizer = optimizer,
            batch_size = batch_size,
        )

        print("it worked!")

        RESULTS = t.train(n_epochs)

    except Exception as e:
        print("Error:", e)
        RESULTS = {"error": str(e)}

    return {
        "RESULTS": RESULTS,
    }