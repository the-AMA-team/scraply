export const TRAINING_DEFAULTS = {
  learningRate: {
    min: 0.001,
    max: 0.1,
    default: 0.001,
    step: 0.001,
  },
  epochs: {
    min: 1,
    max: 1000,
    default: 100,
  },
  batchSize: {
    min: 1,
    max: 100,
    default: 10,
  },
};

/** Max epochs allowed for image datasets (MNIST, FashionMNIST, CIFAR10, etc.) */
export const IMAGE_EPOCHS_MAX = 5;

export const LOSS_FUNCTIONS = [
  { value: "BCE", label: "BCE" },
  { value: "CrossEntropy", label: "CrossEntropy" },
  { value: "BCEWithLogitsLoss", label: "BCE with Logits Loss"}
] as const;

export const OPTIMIZERS = [
  { value: "Adam", label: "Adam" },
  { value: "AdamW", label: "AdamW" },
  { value: "SGD", label: "SGD" },
  { value: "RMSprop", label: "RMSprop" },
] as const;

/** Pima / tabular defaults */
export const DEFAULT_TRAINING_CONFIG = {
  loss: "BCE" as const,
  optimizer: "Adam" as const,
  learningRate: TRAINING_DEFAULTS.learningRate.default,
  epochs: TRAINING_DEFAULTS.epochs.default,
  batchSize: TRAINING_DEFAULTS.batchSize.default,
  runName: "",
};

/** Image dataset defaults (MNIST, FashionMNIST, CIFAR10, …) */
export const IMAGE_TRAINING_DEFAULTS = {
  loss: "CrossEntropy" as const,
  optimizer: "Adam" as const,
  learningRate: TRAINING_DEFAULTS.learningRate.default,
  epochs: 2,
  batchSize: 64,
  runName: "",
};

export function isImageDataset(input: string): boolean {
  return input !== "pima";
}

export function getTrainingDefaultsForDataset(input: string) {
  return isImageDataset(input)
    ? IMAGE_TRAINING_DEFAULTS
    : DEFAULT_TRAINING_CONFIG;
}

export function getMaxEpochsForDataset(input: string): number {
  return isImageDataset(input)
    ? IMAGE_EPOCHS_MAX
    : TRAINING_DEFAULTS.epochs.max;
}

export function getEpochsLimitError(
  input: string,
  epochs: number,
): string | null {
  const max = getMaxEpochsForDataset(input);
  if (epochs > max) {
    return `Max ${max} epochs allowed for this dataset`;
  }
  return null;
}
