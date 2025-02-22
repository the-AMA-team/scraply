# importing the requests library
import requests

# api-endpoint
URL = "http://127.0.0.1:5000/train"

params = {
    "input": "pima",
    "layers": [
        {"kind": "Linear", "args": (8, 12)},
        {"kind": "ReLU"},
        {"kind": "Linear", "args": (12, 8)},
        {"kind": "ReLU"},
        {"kind": "Dropout", "args": 0.2}, # for testing!!!
        {"kind": "Linear", "args": (8, 1)},
        {"kind": "Sigmoid"},
    ],
    "loss": "BCE",
    "optimizer": {"kind": "Adam", "lr": 0.001},
    "epoch": 100,
    "batch_size": 10,
}

try:
    response = requests.post(URL, json=params)
    # Print the response
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except:
    print("url not found")
