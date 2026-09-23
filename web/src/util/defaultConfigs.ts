import { UILayer } from "~/types/index";
import { LAYER_BLOCKS } from "./LAYER_BLOCKS";

// Helper to get label and color from LAYER_BLOCKS by label (case-insensitive)
export function getBlockMeta<L extends string>(label: L) {
  const block = LAYER_BLOCKS.find(
    (b) => b.label.toLowerCase() === label.toLowerCase(),
  );
  return {
    label: (block?.label || label) as L,
    color: block?.color || "#CCCCCC",
  };
}

// Helper to get full block definition from LAYER_BLOCKS by label (case-insensitive)
export function getBlockByLabel(label: string) {
  return LAYER_BLOCKS.find(
    (b) => b.label.toLowerCase() === label.toLowerCase(),
  );
}

// Default layer configurations for each dataset
export const DEFAULT_DATASET_CONFIGS: Record<string, UILayer[]> = {
  // MNIST: 28x28 grayscale images, 10 classes
  MNIST: [
    {
      id: `conv-${Date.now()}-1`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 1,
        outputNeurons: 32,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 0,
      },
    } as const,
    {
      id: `conv-${Date.now()}-2`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 32,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 0,
      },
    } as const,
    {
      id: `maxpool-${Date.now()}`,
      ...getBlockMeta("MaxPool"),
      params: {
        dimension: 2 as 2,
        kernelSize: 2,
        stride: 2,
        padding: 0,
      },
    } as const,
    {
      id: `dropout-${Date.now()}-1`,
      ...getBlockMeta("Dropout"),
      params: {
        dropout: 0.25,
      },
    } as const,
    {
      id: `flatten-${Date.now()}`,
      ...getBlockMeta("Flatten"),
      inputNeurons: 9216,
      startDimension: 1,
      endDimension: -1,
    } as const,
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 9216,
        outputNeurons: 128,
      },
    } as const,
    {
      id: `dropout-${Date.now()}-2`,
      ...getBlockMeta("Dropout"),
      params: {
        dropout: 0.5,
      },
    } as const,
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "No Activation",
      params: {
        inputNeurons: 128,
        outputNeurons: 10,
      },
    } as const,
  ],

  // FashionMNIST: 28x28 grayscale images, 10 classes (similar to MNIST but with different complexity)
  FashionMNIST: [
    {
      id: `conv-${Date.now()}-1`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 1,
        outputNeurons: 32,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `maxpool-${Date.now()}-1`,
      ...getBlockMeta("MaxPool"),
      params: {
        dimension: 2 as 2,
        kernelSize: 2,
        stride: 2,
        padding: 0,
      },
    } as const,
    {
      id: `conv-${Date.now()}-2`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 32,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `maxpool-${Date.now()}-2`,
      ...getBlockMeta("MaxPool"),
      params: {
        dimension: 2 as 2,
        kernelSize: 2,
        stride: 2,
        padding: 0,
      },
    } as const,
    {
      id: `conv-${Date.now()}-3`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 64,
        outputNeurons: 128,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `flatten-${Date.now()}`,
      ...getBlockMeta("Flatten"),
      inputNeurons: 6272,
      startDimension: 1,
      endDimension: -1,
    } as const,
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 6272,
        outputNeurons: 256,
      },
    } as const,
    {
      id: `dropout-${Date.now()}`,
      ...getBlockMeta("Dropout"),
      params: {
        dropout: 0.5,
      },
    } as const,
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "No Activation",
      params: {
        inputNeurons: 256,
        outputNeurons: 10,
      },
    } as const,
  ],

  // CIFAR-10: 32x32 RGB images, 10 classes
  CIFAR10: [
    {
      id: `conv-${Date.now()}-1`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 3,
        outputNeurons: 32,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-2`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 32,
        outputNeurons: 32,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `maxpool-${Date.now()}-1`,
      ...getBlockMeta("MaxPool"),
      params: {
        dimension: 2 as 2,
        kernelSize: 2,
        stride: 2,
        padding: 0,
      },
    } as const,
    {
      id: `dropout-${Date.now()}-1`,
      ...getBlockMeta("Dropout"),
      params: {
        dropout: 0.25,
      },
    } as const,
    {
      id: `conv-${Date.now()}-3`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 32,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-4`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 64,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `maxpool-${Date.now()}-2`,
      ...getBlockMeta("MaxPool"),
      params: {
        dimension: 2 as 2,
        kernelSize: 2,
        stride: 2,
        padding: 0,
      },
    } as const,
    {
      id: `dropout-${Date.now()}-2`,
      ...getBlockMeta("Dropout"),
      params: {
        dropout: 0.25,
      },
    } as const,
    {
      id: `flatten-${Date.now()}`,
      ...getBlockMeta("Flatten"),
      inputNeurons: 4096,
      startDimension: 1,
      endDimension: -1,
    } as const,
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 4096,
        outputNeurons: 512,
      },
    } as const,
    {
      id: `dropout-${Date.now()}-3`,
      ...getBlockMeta("Dropout"),
      params: {
        dropout: 0.5,
      },
    } as const,
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "No Activation",
      params: {
        inputNeurons: 512,
        outputNeurons: 10,
      },
    } as const,
  ],

  // Pima Diabetes: 8 features, 2 classes (binary classification)
  pima: [
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 8,
        outputNeurons: 10,
      },
    } as const,
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 10,
        outputNeurons: 8,
      },
    } as const,
    {
      id: `linear-${Date.now()}-3`,
      ...getBlockMeta("Linear"),
      activationFunction: "Sigmoid",
      params: {
        inputNeurons: 8,
        outputNeurons: 1,
      },
    } as const,
  ],
};

// LeNet-style default configs for each dataset (classic small CNN)
// Note: LeNet is primarily for 28x28 grayscale datasets; for CIFAR10 we adapt channels.
export const LENET_DATASET_CONFIGS: Record<string, UILayer[]> = {
  MNIST: [
    {
      id: `conv-${Date.now()}-1`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 1,
        outputNeurons: 6,
        dimension: 2 as 2,
        kernelSize: 5,
        stride: 1,
        padding: 0,
      },
    },
    {
      id: `avgpool-${Date.now()}-1`,
      ...getBlockMeta("AvgPool"),
      params: { dimension: 2 as 2, kernelSize: 2, stride: 2, padding: 0 },
    },
    {
      id: `conv-${Date.now()}-2`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 6,
        outputNeurons: 16,
        dimension: 2 as 2,
        kernelSize: 5,
        stride: 1,
        padding: 0,
      },
    },
    {
      id: `avgpool-${Date.now()}-2`,
      ...getBlockMeta("AvgPool"),
      params: { dimension: 2 as 2, kernelSize: 2, stride: 2, padding: 0 },
    },
    {
      id: `flatten-${Date.now()}`,
      ...getBlockMeta("Flatten"),
      // LeNet spatial math on 28x28: conv5 -> 24x24 -> pool -> 12x12 -> conv5 -> 8x8 -> pool -> 4x4
      inputNeurons: 16 * 4 * 4, // = 256
      startDimension: 1,
      endDimension: -1,
    },
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: { inputNeurons: 256, outputNeurons: 120 },
    },
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: { inputNeurons: 120, outputNeurons: 84 },
    },
    {
      id: `linear-${Date.now()}-3`,
      ...getBlockMeta("Linear"),
      activationFunction: "No Activation",
      params: { inputNeurons: 84, outputNeurons: 10 },
    },
  ],

  FashionMNIST: [
    // Identical shape to MNIST (28x28 grayscale) — same LeNet config
    {
      id: `conv-${Date.now()}-1`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 1,
        outputNeurons: 6,
        dimension: 2 as 2,
        kernelSize: 5,
        stride: 1,
        padding: 0,
      },
    },
    {
      id: `avgpool-${Date.now()}-1`,
      ...getBlockMeta("AvgPool"),
      params: { dimension: 2 as 2, kernelSize: 2, stride: 2, padding: 0 },
    },
    {
      id: `conv-${Date.now()}-2`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 6,
        outputNeurons: 16,
        dimension: 2 as 2,
        kernelSize: 5,
        stride: 1,
        padding: 0,
      },
    },
    {
      id: `avgpool-${Date.now()}-2`,
      ...getBlockMeta("AvgPool"),
      params: { dimension: 2 as 2, kernelSize: 2, stride: 2, padding: 0 },
    },
    {
      id: `flatten-${Date.now()}`,
      ...getBlockMeta("Flatten"),
      inputNeurons: 16 * 4 * 4, // 256
      startDimension: 1,
      endDimension: -1,
    },
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: { inputNeurons: 256, outputNeurons: 120 },
    },
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: { inputNeurons: 120, outputNeurons: 84 },
    },
    {
      id: `linear-${Date.now()}-3`,
      ...getBlockMeta("Linear"),
      activationFunction: "No Activation",
      params: { inputNeurons: 84, outputNeurons: 10 },
    },
  ],

  CIFAR10: [
    // LeNet adapted for 32x32 RGB: increase channels slightly
    {
      id: `conv-${Date.now()}-1`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 3,
        outputNeurons: 16,
        dimension: 2 as 2,
        kernelSize: 5,
        stride: 1,
        padding: 0,
      },
    },
    {
      id: `avgpool-${Date.now()}-1`,
      ...getBlockMeta("AvgPool"),
      params: { dimension: 2 as 2, kernelSize: 2, stride: 2, padding: 0 },
    },
    {
      id: `conv-${Date.now()}-2`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 16,
        outputNeurons: 32,
        dimension: 2 as 2,
        kernelSize: 5,
        stride: 1,
        padding: 0,
      },
    },
    {
      id: `avgpool-${Date.now()}-2`,
      ...getBlockMeta("AvgPool"),
      params: { dimension: 2 as 2, kernelSize: 2, stride: 2, padding: 0 },
    },
    {
      id: `flatten-${Date.now()}`,
      ...getBlockMeta("Flatten"),
      // 32x32 conv5 -> 28x28 -> pool -> 14x14 -> conv5 -> 10x10 -> pool -> 5x5
      inputNeurons: 32 * 5 * 5, // = 800
      startDimension: 1,
      endDimension: -1,
    },
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: { inputNeurons: 800, outputNeurons: 120 },
    },
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: { inputNeurons: 120, outputNeurons: 84 },
    },
    {
      id: `linear-${Date.now()}-3`,
      ...getBlockMeta("Linear"),
      activationFunction: "No Activation",
      params: { inputNeurons: 84, outputNeurons: 10 },
    },
  ],
};

// ResNet-style default configs for each dataset (modern residual blocks — small-image adapted)
// Represented in UI as conv stacks + downsampling convs (stride=2) and final global pool -> FC.
export const RESNET_DATASET_CONFIGS: Record<string, UILayer[]> = {
  MNIST: [
    // Small-image ResNet pattern for 28x28 grayscale — first conv 3x3 stride1
    {
      id: `conv-${Date.now()}-1`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 1,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    // Res block stage 1 (2 convs)
    {
      id: `conv-${Date.now()}-2`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 64,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-3`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 64,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    // Downsample stage (simulate stride-2 res block start)
    {
      id: `conv-${Date.now()}-4`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 64,
        outputNeurons: 128,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 2,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-5`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 128,
        outputNeurons: 128,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    // Further downsampling
    {
      id: `conv-${Date.now()}-6`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 128,
        outputNeurons: 256,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 2,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-7`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 256,
        outputNeurons: 256,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    // Final flatten (spatial size is 7x7 after two stride-2 downsamples)
    {
      id: `flatten-${Date.now()}`,
      ...getBlockMeta("Flatten"),
      inputNeurons: 256 * 7 * 7, // 12544
      startDimension: 1,
      endDimension: -1,
    } as const,
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: { inputNeurons: 256 * 7 * 7, outputNeurons: 512 }, // 12544 -> 512
    } as const,
    {
      id: `dropout-${Date.now()}-1`,
      ...getBlockMeta("Dropout"),
      params: { dropout: 0.5 },
    } as const,
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "No Activation",
      params: { inputNeurons: 512, outputNeurons: 10 },
    } as const,
  ],

  FashionMNIST: [
    // Same structure as MNIST but first conv inputNeurons = 1
    {
      id: `conv-${Date.now()}-1`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 1,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-2`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 64,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-3`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 64,
        outputNeurons: 128,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 2,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-4`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 128,
        outputNeurons: 128,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-5`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 128,
        outputNeurons: 256,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 2,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-6`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 256,
        outputNeurons: 256,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `flatten-${Date.now()}`,
      ...getBlockMeta("Flatten"),
      inputNeurons: 256 * 7 * 7, // 12544
      startDimension: 1,
      endDimension: -1,
    } as const,
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: { inputNeurons: 256 * 7 * 7, outputNeurons: 512 }, // 12544 -> 512
    } as const,
    {
      id: `dropout-${Date.now()}-1`,
      ...getBlockMeta("Dropout"),
      params: { dropout: 0.5 },
    } as const,
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "No Activation",
      params: { inputNeurons: 512, outputNeurons: 10 },
    } as const,
  ],

  CIFAR10: [
    // ResNet-18-ish small-image adaptation for CIFAR10
    {
      id: `conv-${Date.now()}-1`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 3,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    // stage 1
    {
      id: `conv-${Date.now()}-2`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 64,
        outputNeurons: 64,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-3`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 64,
        outputNeurons: 128,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 2,
        padding: 1,
      },
    } as const,
    // stage 2
    {
      id: `conv-${Date.now()}-4`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 128,
        outputNeurons: 128,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-5`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 128,
        outputNeurons: 256,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 2,
        padding: 1,
      },
    } as const,
    // stage 3
    {
      id: `conv-${Date.now()}-6`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 256,
        outputNeurons: 256,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    {
      id: `conv-${Date.now()}-7`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 256,
        outputNeurons: 512,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 2,
        padding: 1,
      },
    } as const,
    // final stage conv
    {
      id: `conv-${Date.now()}-8`,
      ...getBlockMeta("Conv"),
      activationFunction: "ReLU",
      params: {
        inputNeurons: 512,
        outputNeurons: 512,
        dimension: 2 as 2,
        kernelSize: 3,
        stride: 1,
        padding: 1,
      },
    } as const,
    // final flatten (spatial size is 4x4 after stride-2 downsamples)
    {
      id: `flatten-${Date.now()}`,
      ...getBlockMeta("Flatten"),
      inputNeurons: 512 * 4 * 4, // 8192
      startDimension: 1,
      endDimension: -1,
    } as const,
    {
      id: `linear-${Date.now()}-1`,
      ...getBlockMeta("Linear"),
      activationFunction: "ReLU",
      params: { inputNeurons: 512 * 4 * 4, outputNeurons: 512 }, // 8192 -> 512
    } as const,
    {
      id: `dropout-${Date.now()}-1`,
      ...getBlockMeta("Dropout"),
      params: { dropout: 0.5 },
    } as const,
    {
      id: `linear-${Date.now()}-2`,
      ...getBlockMeta("Linear"),
      activationFunction: "No Activation",
      params: { inputNeurons: 512, outputNeurons: 10 },
    } as const,
  ],
};

const withFreshIds = (blocks: UILayer[]): UILayer[] => {
  const timestamp = Date.now();
  return blocks.map((block, index) => ({
    ...block,
    id: `${block.label.toLowerCase()}-${timestamp}-${index}`,
  }));
};

export const generateUniqueBlocks = (datasetName: string): UILayer[] => {
  const baseConfig = DEFAULT_DATASET_CONFIGS[datasetName];
  if (!baseConfig) return [];
  return withFreshIds(baseConfig);
};

export const generateResNetBlocks = (datasetName: string): UILayer[] => {
  const baseConfig = RESNET_DATASET_CONFIGS[datasetName];
  if (!baseConfig) return [];
  return withFreshIds(baseConfig);
};

export const generateLeNetBlocks = (datasetName: string): UILayer[] => {
  const baseConfig = LENET_DATASET_CONFIGS[datasetName];
  if (!baseConfig) return [];
  return withFreshIds(baseConfig);
};

// Dataset-wise architecture availability mapping
export const DATASET_ARCHITECTURES: Record<string, string[]> = {
  pima: ["custom", "default"],
  MNIST: ["custom", "default", "lenet", "resnet"],
  FashionMNIST: ["custom", "default", "lenet", "resnet"],
  CIFAR10: ["custom", "default", "lenet", "resnet"],
};

// Helper function to get available architectures for a dataset
export const getAvailableArchitectures = (datasetName: string): string[] => {
  return DATASET_ARCHITECTURES[datasetName] || ["custom", "default"];
};
