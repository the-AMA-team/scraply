from flask import Flask, request
from models import DynamicModel, Train
from params import are_params_valid
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world():
    return {"data": "hello"}


@app.post("/train")
def train():
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
    t = Train(model, inp, loss, optimizer)
    f_l = t.train(n_epochs, batch_size)

    return {"final_loss": f_l.item()}
