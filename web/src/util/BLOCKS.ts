import * as tf from "@tensorflow/tfjs";
import { MappedBlocks } from "~/types";

export const BLOCKS: MappedBlocks = [
  {
    id: "linear",
    label: "Linear",
    color: "#20FF8F",
    activationFunction: tf.layers.reLU(),
    tfFunctionArgs: [8, [8]],
    tfFunction: (units, inputShape) => tf.layers.dense({ units, inputShape }),
  },
  {
    id: "conv",
    label: "Conv",
    color: "#FFD620",
    activationFunction: tf.layers.reLU(),
    tfFunctionArgs: [8, 3, [8]],
    tfFunction: (filters, kernelSize, inputShape) =>
      tf.layers.conv1d({ filters, kernelSize, inputShape }),
  },
  {
    id: "rnn",
    label: "RNN",
    color: "#FF8C20",
    activationFunction: tf.layers.reLU(),
    tfFunctionArgs: [tf.layers.simpleRNNCell({ units: 8 }), [8]],
    tfFunction: (cell, inputShape) => tf.layers.rnn({ cell, inputShape }),
  },
  {
    id: "gru",
    label: "GRU",
    color: "#FF4920",
    activationFunction: tf.layers.reLU(),
    tfFunctionArgs: [8, [8]],
    tfFunction: (units, inputShape) => tf.layers.gru({ units, inputShape }),
  },
  {
    id: "flatten",
    label: "Flatten",
    color: "#FF208F",
    activationFunction: null,
    tfFunctionArgs: [[8]],
    tfFunction: (inputShape) => tf.layers.flatten({ inputShape }),
  },
];
