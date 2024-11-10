from flask import Flask, request, send_file
from models import DynamicModel, Train
from params import are_params_valid
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
    #     data = {
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
    inp = data["input"]
    layers = data["layers"]
    loss = data["loss"]
    optimizer = data["optimizer"]
    n_epochs = data["epoch"]
    batch_size = data["batch_size"]

    if not are_params_valid(data):
        return {"error": "Invalid parameters"}

    model = DynamicModel(layers)
    # print(model)

    t = Train(
        model=model,
        input=inp,
        loss=loss,
        optimizer=optimizer,
        batch_size=batch_size,
    )

    RESULTS = t.train_test_log(n_epochs, batch_size)

    # print("Results:")
    # print("Average Training Loss: ", RESULTS["avg_train_loss"])
    # print("Average Testing Loss: ", RESULTS["avg_test_loss"])
    # print("Average Training Accuracy: ", RESULTS["avg_train_acc"])
    # print("Average Testing Accuracy: ", RESULTS["avg_test_acc"])

    return RESULTS


t = {
    "layers": [
        {"kind": "Linear", "args": [8, 12]},
        {"kind": "ReLU"},
        {"kind": "Linear", "args": [12, 8]},
        {"kind": "ReLU"},
        {"kind": "Linear", "args": [8, 1]},
        {"kind": "Sigmoid"},
    ],
    "loss": "BCE",
    "optimizer": {"kind": "Adam", "lr": 0.001},
    "epoch": 100,
    "batch_size": 10,
}
