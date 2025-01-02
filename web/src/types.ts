export interface Layer {
  id: string;
  label: string;
  color: string;
  neurons: number;
  activationFunction: string;
}

export interface ActivationFunction {
  kind: "RerU" | "Sigmoid" | "Tanh" | "Softmax" | "LeakyReLU" | "PReLU";
}

export type Block = Layer | ActivationFunction;
