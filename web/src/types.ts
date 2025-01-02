export interface UILayer {
  id: string;
  label: string;
  color: string;
  neurons: number;
  activationFunction: ActivationFunction;
}

export type ActivationFunction =
  | "ReLU"
  | "Sigmoid"
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
