import * as tf from "@tensorflow/tfjs";
import { UILayer } from "~/types";

type MappedBlocks = [
  UILayer<(units: number, inputShape: number[]) => tf.layers.Layer>,
  UILayer<
    (
      filters: number,
      kernelSize: number,
      inputShape: number[],
    ) => tf.layers.Layer
  >,
  UILayer<(cell: tf.layers.RNNCell, inputShape: number[]) => tf.layers.Layer>,
  UILayer<(units: number, inputShape: number[]) => tf.layers.Layer>,
  UILayer<(inputShape: number[]) => tf.layers.Layer, null>,
];

export const BLOCKS: MappedBlocks = [
  {
    id: "linear",
    label: "Linear",
    color: "#20FF8F",
    activationFunction: tf.layers.reLU(),
    neurons: 8,
    tfFunction: (units, inputShape) => tf.layers.dense({ units, inputShape }),
  },
  {
    id: "conv",
    label: "Conv",
    color: "#FFD620",
    activationFunction: tf.layers.reLU(),
    neurons: 8,
    tfFunction: (filters, kernelSize, inputShape) =>
      tf.layers.conv1d({ filters, kernelSize, inputShape }),
  },
  {
    id: "rnn",
    label: "RNN",
    color: "#FF8C20",
    activationFunction: tf.layers.reLU(),
    neurons: 8,
    tfFunction: (cell, inputShape) => tf.layers.rnn({ cell, inputShape }),
  },
  {
    id: "gru",
    label: "GRU",
    color: "#FF4920",
    activationFunction: tf.layers.reLU(),
    neurons: 8,
    tfFunction: (units, inputShape) => tf.layers.gru({ units, inputShape }),
  },
  {
    id: "flatten",
    label: "Flatten",
    color: "#FF208F",
    activationFunction: null,
    neurons: 8,
    tfFunction: (inputShape) => tf.layers.flatten({ inputShape }),
  },
];
