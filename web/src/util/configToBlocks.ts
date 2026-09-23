import { Config, UILayer, ActivationFunction, hasActivationFunction } from "~/types/index";
import { getBlockMeta, getBlockByLabel } from "./defaultConfigs";

// Convert Config layers back to UILayer blocks for display
export function configToBlocks(config: Config): UILayer[] {
  const blocks: UILayer[] = [];
  let blockIdCounter = 0;

  for (let i = 0; i < config.layers.length; i++) {
    const layer = config.layers[i];
    
    // Skip if layer is undefined or if it's an activation function
    if (!layer || !("args" in layer)) {
      continue;
    }

    // TypeScript guard: layer is now guaranteed to have 'args'
    const layerWithArgs = layer as { kind: string; args: number[] };
    const kind = layerWithArgs.kind;
    const args = layerWithArgs.args;

    // Determine the base label (handle Conv1D/Conv2D -> Conv, etc.)
    // This mirrors the logic in board.util.ts getConfig function
    let baseLabel = kind;
    if (kind.startsWith("Conv")) {
      baseLabel = "Conv";
    } else if (kind.startsWith("MaxPool")) {
      baseLabel = "MaxPool";
    } else if (kind.startsWith("AvgPool")) {
      baseLabel = "AvgPool";
    }

    // Get the block definition from LAYER_BLOCKS
    const blockTemplate = getBlockByLabel(baseLabel);
    if (!blockTemplate) {
      // Skip unknown layer types
      continue;
    }

    const blockId = `${baseLabel.toLowerCase()}-${blockIdCounter++}`;
    const meta = getBlockMeta(baseLabel);

    // Handle Linear layers - args: [inputNeurons, outputNeurons]
    if (baseLabel === "Linear") {
      blocks.push({
        id: blockId,
        label: "Linear",
        color: meta.color,
        activationFunction: getNextActivation(config.layers, i, blockTemplate),
        params: {
          inputNeurons: args[0]!,
          outputNeurons: args[1]!,
        },
      });
    }
    // Handle Conv layers - args: [dimension, inputChannels, outputChannels, kernelSize, stride, padding]
    else if (baseLabel === "Conv") {
      blocks.push({
        id: blockId,
        label: "Conv",
        color: meta.color,
        activationFunction: getNextActivation(config.layers, i, blockTemplate),
        params: {
          dimension: args[0] as 1 | 2,
          inputNeurons: args[1]!,
          outputNeurons: args[2]!,
          kernelSize: args[3]!,
          stride: args[4]!,
          padding: args[5]!,
        },
      });
    }
    // Handle MaxPool layers - args: [dimension, kernelSize, stride, padding]
    else if (baseLabel === "MaxPool") {
      blocks.push({
        id: blockId,
        label: "MaxPool",
        color: meta.color,
        params: {
          dimension: args[0] as 1 | 2,
          kernelSize: args[1]!,
          stride: args[2]!,
          padding: args[3]!,
        },
      });
    }
    // Handle AvgPool layers - args: [dimension, kernelSize, stride, padding]
    else if (baseLabel === "AvgPool") {
      blocks.push({
        id: blockId,
        label: "AvgPool",
        color: meta.color,
        params: {
          dimension: args[0] as 1 | 2,
          kernelSize: args[1]!,
          stride: args[2]!,
          padding: args[3]!,
        },
      });
    }
    // Handle Dropout layers - args: [dropout]
    else if (kind === "Dropout") {
      blocks.push({
        id: blockId,
        label: "Dropout",
        color: meta.color,
        params: {
          dropout: args[0]!,
        },
      });
    }
    // Handle Flatten layers - no args typically
    else if (kind === "Flatten") {
      // Use the template from LAYER_BLOCKS for default values
      const flattenTemplate = getBlockByLabel("Flatten");
      blocks.push({
        id: blockId,
        label: "Flatten",
        color: meta.color,
        inputNeurons: flattenTemplate && "inputNeurons" in flattenTemplate 
          ? flattenTemplate.inputNeurons 
          : 0,
        startDimension: flattenTemplate && "startDimension" in flattenTemplate
          ? flattenTemplate.startDimension
          : 1,
        endDimension: flattenTemplate && "endDimension" in flattenTemplate
          ? flattenTemplate.endDimension
          : -1,
      });
    }
    // Handle RNN layers - args: [inputSize, hiddenSize, dropout]
    else if (kind === "RNN") {
      const rnnTemplate = getBlockByLabel("RNN");
      blocks.push({
        id: blockId,
        label: "RNN",
        color: meta.color,
        activationFunction: getNextActivation(config.layers, i, rnnTemplate),
        params: {
          inputNeurons: args[0]!,
          outputNeurons: args[1]!, // hiddenSize
          hiddenSize: args[1]!,
          dropout: args[2] || 0,
        },
      });
    }
    // Handle GRU layers - args: [inputSize, hiddenSize, dropout]
    else if (kind === "GRU") {
      const gruTemplate = getBlockByLabel("GRU");
      blocks.push({
        id: blockId,
        label: "GRU",
        color: meta.color,
        activationFunction: getNextActivation(config.layers, i, gruTemplate),
        params: {
          inputNeurons: args[0]!,
          outputNeurons: args[1]!, // hiddenSize
          hiddenSize: args[1]!,
          dropout: args[2] || 0,
        },
      });
    }
  }

  return blocks;
}

// Helper to get the next activation function from layers array
// Uses the block template's default activation function if no explicit activation is found
function getNextActivation(
  layers: Config["layers"],
  currentIndex: number,
  blockTemplate?: UILayer,
): ActivationFunction {
  const nextLayer = layers[currentIndex + 1];
  if (nextLayer && !("args" in nextLayer)) {
    const activation = nextLayer.kind;
    // Validate that it's a valid activation function
    const validActivations: ActivationFunction[] = [
      "ReLU",
      "Sigmoid",
      "Tanh",
      "Softmax",
      "LeakyReLU",
      "PReLU",
      "No Activation",
    ];
    if (validActivations.includes(activation as ActivationFunction)) {
      return activation as ActivationFunction;
    }
  }
  // Use default from block template if available, otherwise default to ReLU
  if (blockTemplate && hasActivationFunction(blockTemplate)) {
    return blockTemplate.activationFunction;
  }
  return "ReLU";
}
