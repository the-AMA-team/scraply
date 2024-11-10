IDEAL_ARCH = [
    {
        "layers": [
            {"kind": "Linear", "args": (8, 12)},
            {"kind": "ReLU"},
            {"kind": "Linear", "args": (12, 8)},
            {"kind": "ReLU"},
            {"kind": "Linear", "args": (8, 1)},
            {"kind": "Sigmoid"},
        ],
        "loss": "BCE",
        "optimizer": {"kind": "Adam", "lr": 0.001},
        "epoch": 100,
        "batch_size": 10,
    }
]
