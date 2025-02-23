export const systemPrompt = () => `
  For the dataset provided by the user, write out a archetecture as JSON.
  The JSON shape should be of the following format:

    {
      layers: [
        { kind: "Linear", args: (8, 12) },
        { kind: "ReLU" },
        { kind: "Linear", args: (12, 8) },
        { kind: "ReLU" },
        { kind: "Linear", args: (8, 1) },
        { kind: "Sigmoid" },
      ],
      loss: "BCE",
      optimizer: { kind: "Adam", lr: 0.001 },
      epoch: 100,
      batch_size: 10,
    }
  
    You are allowed to choose from the following parameters:
    layers: "Linear", "Conv1D", "Conv2D", "Conv3D", "LSTM", "GRU", "RNN", "Flatten", "ReLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU", "PReLU"
    loss: "BCE", "CrossEntropy"
    optimizer: "Adam", "AdamW", "SGD", "RMSprop"

    1. activation functions are also written as layers but without any neuron arguments.
    2. The first argument of the Linear layer is the number of input neurons and the second argument is the number of output neurons.
    3. The first argument of the Conv2D layer is the number of input channels, the second argument is the number of output channels, and the third argument is the kernel size.
    4. The first argument of the LSTM layer is the number of input neurons and the second argument is the number of hidden neurons.
    5. The first argument of the GRU layer is the number of input neurons and the second argument is the number of hidden neurons.
    6. The first argument of the RNN layer is the number of input neurons and the second argument is the number of hidden neurons.
    7. The first argument of the Flatten layer is the number of input neurons.

    RETURN A VALID JSON PARSABLE STRING!!! DO NOT USE THE JSON INDICATOR \`\`\`json\`\`\ AT THE BEGINING!!!
`;

export const userPrompt = (dataset: string) => {
  return `Get suggestions for the ${dataset} dataset`;
};
