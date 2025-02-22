from flask import Flask, request, send_file
from models import DynamicModel, Train
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
    }
