import { UILayer } from "~/types";

export const LAYER_BLOCKS: UILayer[] = [
  {
    id: "linear",
    label: "Linear",
    color: "#20FF8F",
    activationFunction: "ReLU",
    neurons: 8,
  },
  {
    id: "conv",
    label: "Conv",
    color: "#FFD620",
    activationFunction: "ReLU",
    neurons: 8,
  },
  {
    id: "rnn",
    label: "RNN",
    color: "#FF8C20",
    activationFunction: "ReLU",
    neurons: 8,
  },
  {
    id: "gru",
    label: "GRU",
    color: "#FF4920",
    activationFunction: "ReLU",
    neurons: 8,
  },
  {
    id: "flatten",
    label: "Flatten",
    color: "#FF208F",
    activationFunction: "",
    neurons: 8,
  },
];
