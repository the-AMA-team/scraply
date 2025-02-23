from flask import Flask, request, send_file
from models import DynamicModel, TransformerModel, TransformerTrain, Train
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

@app.route("/inference", methods=["POST"])
def inference():
    infer_args = request.get_json()
    
    # example infer_args
    infer_args = {
        "temperature": 0.1,
        "generation length": 100,
        "input": "Alice was"
    }

    try:
        # inference method here
        print("hello")
        
    except Exception as e:
        return {"status": "failed (mehek messed up)", "error": str(e)}


@app.post("/train")
def train():
    # example arguments
    embed_dim = 100
    heads = 2
    hidden_dim = 2048
    # example data
    data = {
        "type": "transformer", # ADDED NEW PARAMETER
        "input": "alice", # preprocess
        "layers": [
            {"kind": "Decoder", "args": (embed_dim, heads, hidden_dim)},
            {"kind": "Decoder", "args": (embed_dim, heads, hidden_dim)},
            {"kind": "Output", "args": (12, 8)},
        ],
        "loss": "BCE",
        "optimizer": {"kind": "Adam", "lr": 0.001},
        "epoch": 100,
        "batch_size": 10,
    }

    #data = request.get_json()
    print("Received data:", data)
    type = data["type"]

    inp = data["input"]
    layers = data["layers"]
    loss = data["loss"]
    optimizer = data["optimizer"]
    n_epochs = data["epoch"]
    batch_size = data["batch_size"]

    RESULTS = {}

    try:
        if(type == "transformers"):
            model = TransformerModel(layers)
            
            t = TransformerTrain(
                model=model,
                input=inp,
                loss=loss,
                optimizer=optimizer,
                batch_size=batch_size,
            )
            
            print("TRANSFORMER SLAYYY")
            
            RESULTS = t.train_test_log(n_epochs, batch_size)
        else: 
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
        print("Error (mehek messed up):", e)
        RESULTS = {"error (mehek messed up)": str(e)}

    return {
        "RESULTS": RESULTS,
    }
