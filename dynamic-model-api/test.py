# importing the requests library
import requests

# api-endpoint
URL = "http://127.0.0.1:5000/transformertrain"

# example arguments
embed_dim = 10
heads = 2
hidden_dim = 2048
# example data
params = {
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

try:
    response = requests.post(URL, json=params)
    # Print the response
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except:
    print("url not found")
