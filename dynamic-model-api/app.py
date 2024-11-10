from flask import Flask, request, send_file
from models import DynamicModel, Train
from params import are_params_valid
from IdealArch import IDEAL_ARCH
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

    # if not are_params_valid(data):
    #     return {"error": "Invalid parameters"}

    size = len(layers)
    Lkind_advice_array = [""] * size  # Creates an array of size 'layers'
    Largs_advice_array = [""] * size  # Creates an array of size 'layers'
    lastLayer = ""

    # CHECK IF THE LAST LAYER IS A SIGMOID OR SOFTMAX

    # print("slay")

    if layers[-1]["kind"] not in ("Sigmoid", "Softmax"):
        lastLayer = "The last activation needs to be a Sigmoid or Softmax function. Other activation functions can lead to unstable predictions"
        Lkind_advice_array[-1] = lastLayer
    else:
        Lkind_advice_array[-1] = ""

    print(Lkind_advice_array)

    # CHECK LAYER TYPES AGAINST IDEAL ARCHITECTURE
    for i in range(len(layers) - 1):
        # compare kind to ideal arch
        if layers[i]["kind"] != IDEAL_ARCH[0]["layers"][i]["kind"]:
            Lkind_advice_array[i] = (
                "The current layer is not the ideal layer. Consider changing the current layer to match the ideal layer."
            )
        else:
            Lkind_advice_array[i] = ""

    # CHECK LAYER ARGUMENTS MATCH BETWEEN LAYERS
    for i in range(len(layers) - 2):
        if layers[i]["kind"] == "Linear" and layers[i + 2]["kind"] == "Linear":
            if layers[i]["args"][1] != layers[i + 2]["args"][0]:
                Largs_advice_array[i] = (
                    "The output of the current layer does not match the input of the next layer. Consider changing the output of the current layer to match the input of the next layer."
                )
            else:
                Largs_advice_array[i] = ""

    loss_str = ""
    optimizer_str = ""
    epoch_str = ""
    batch_size_str = ""

    # CHECK IF LOSS, OPTIMIZER, EPOCH, BATCH SIZE MATCH IDEAL ARCHITECTURE
    for key in data:
        if key == "layers":
            continue  # Skip the layers keys, should be already filled
        else:
            if key == "loss":
                if data[key] != IDEAL_ARCH[0][key]:
                    loss_str = f"Current loss function is {data[key]}, consider changing to {IDEAL_ARCH[0][key]}."
                else:
                    loss_str = ""
            elif key == "optimizer":
                if data[key] != IDEAL_ARCH[0][key]:
                    optimizer_str = f"Current optimizer is {data[key]}, consider changing to {IDEAL_ARCH[0][key]}."
            elif key == "epoch":
                if data[key] != IDEAL_ARCH[0][key]:
                    epoch_str = f"Current epoch is {data[key]}, consider changing to {IDEAL_ARCH[0][key]}."
            elif key == "batch_size":
                if data[key] != IDEAL_ARCH[0][key]:
                    batch_size_str = f"Current batch size is {data[key]}, consider changing to {IDEAL_ARCH[0][key]}."

    # FILL IN ADVICE DICTIONARY WITH OBTAINED STRINGS
    ADVICE = {
        "layers": [
            {"kind": Lkind_advice_array},
            {"args": Largs_advice_array},
        ],
        "loss": loss_str,
        "optimizer": optimizer_str,
        "epoch": epoch_str,
        "batch_size": batch_size_str,
    }
    # layers is a list of dictionaries
    # layer key in dictionary outputs 1 suggestion string per layer/activation

    print(ADVICE)

    sad_advice_string = ""
    sad_advice_string = ADVICE["layers"][-1]
    print(sad_advice_string)

    RESULTS = {}

    try:
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
    except Exception as e:
        print("Error:", e)
        RESULTS = {"error": str(e)}

    return RESULTS, ADVICE
