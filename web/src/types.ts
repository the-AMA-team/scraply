import * as tf from "@tensorflow/tfjs";

// A defaults to tf.layers.Layer just for the sake of FlattenLayer
export interface UILayer<L extends (...args: any) => any, A = tf.layers.Layer> {
  id: string;
  label: string;
  color: string;
  tfFunctionArgs: Parameters<L>;
  activationFunction: A;
  tfFunction: L;
}

export type LinearLayer = (
  units: number,
  inputShape: number[],
) => tf.layers.Layer;

export type ConvLayer = (
  filters: number,
  kernelSize: number,
  inputShape: number[],
) => tf.layers.Layer;

export type RNNLayer = (
  cell: tf.layers.RNNCell,
  inputShape: number[],
) => tf.layers.Layer;

export type GRULayer = (units: number, inputShape: number[]) => tf.layers.Layer;

export type FlattenLayer = (inputShape: number[]) => tf.layers.Layer;

export type AnyUILayer =
  | UILayer<LinearLayer>
  | UILayer<ConvLayer>
  | UILayer<RNNLayer>
  | UILayer<GRULayer>
  | UILayer<FlattenLayer, null>;

export type MappedBlocks = [
  UILayer<LinearLayer>,
  UILayer<ConvLayer>,
  UILayer<RNNLayer>,
  UILayer<GRULayer>,
  UILayer<FlattenLayer, null>,
];

// convert to enum
export type ActivationFunction =
  | ""
  | "ReLU"
  | "Tanh"
  | "Softmax"
  | "LeakyReLU"
  | "PReLU";

interface ConfigLayer {
  kind: string;
  args: number[];
}

interface ConfigActivationFunction {
  kind: string;
}

type ConfigBlock = ConfigLayer | ConfigActivationFunction;

export interface Config {
  input: string;
  layers: ConfigBlock[];
  loss: string;
  optimizer: {
    kind: string;
    lr: number;
  };
  epoch: number;
  batch_size: number;
}

export enum DataMode {
  UPLOAD = "UPLOAD",
  PRESET = "PRESET",
}

export enum AppMode {
  DATA = "DATA",
  LAYERS = "LAYERS",
}
