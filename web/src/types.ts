export type Dataset = {
  label: string;
  inputName: string;
};

export interface UILayer {
  id: string;
  label: string;
  color: string;
  neurons: number;
  otherParam?: number;
  activationFunction: ActivationFunction;
}

export type ActivationFunction =
  | ""
  | "ReLU"
  | "Sigmoid"
  | "Tanh"
  | "Softmax"
  | "LeakyReLU"
  | "PReLU";

interface Layer {
  kind: string;
  args: number[];
}

interface LayerActivationFunction {
  kind: string;
}

type LayerBlock = Layer | LayerActivationFunction;

export interface Config {
  input: string;
  layers: LayerBlock[];
  loss: string;
  optimizer: {
    kind: string;
    lr: number;
  };
  learning_rate: number;
  epoch: number;
  batch_size: number;
}

export interface TransformerConfig {
  input: string;
  layers: (
    | {
        kind: string;
        args: number[];
      }
    | {
        kind: string;
        args: number;
      }
  )[];
  loss: string;
  optimizer: {
    kind: string;
    lr: number;
  };
  epoch: number;
  batch_size: number;
}

export enum AppMode {
  LAYERS = "LAYERS",
  TRANSFORMERS = "TRANSFORMERS",
}
