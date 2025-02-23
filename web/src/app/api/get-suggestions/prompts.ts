export const systemPrompt = () => `
  For the dataset provided by the user, write out a archetecture as JSON.
  The JSON shape should be of the following format:
    
    Array<{
      label: string;
      color: string;
      neurons: number;
      otherParam?: number;
      activationFunction: 
      | ""
      | "ReLU"
      | "Sigmoid"
      | "Tanh"
      | "Softmax"
      | "LeakyReLU"
      | "PReLU";
    }>

    label: layerKind choosen ONLY from the allowed layers below
    color: choose from: [{label: "Linear", color: "#20FF8F"}, {label: "Conv1D", color: "#FFD620"}, {label: "Conv2D", color: "#FFD620"}, {label: "Conv3D", color: "#FFD620"}, {label: "RNN", color: "#FF8C20"}, {label: "GRU", color: "#FF4920"}, {label: "Flatten", color: "#FF208F"}]
    neurons: number of INPUT neurons (note that the output neurons will just be infered from the input neurons of the next layer)
    otherParam: the kernel size for Conv1D, Conv2D, Conv3D, and the hidden layer size for RNN, GRU 
    activationFunction: activation function choosen ONLY from the allowed activation functions
  
    You are allowed to choose from the following parameters:
    layers: "Linear", "Conv1D", "Conv2D", "Conv3D", "LSTM", "GRU", "RNN", "Flatten", "ReLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU", "PReLU"
    loss: "BCE", "CrossEntropy"
    optimizer: "Adam", "AdamW", "SGD", "RMSprop"

    The JSON must be an array of objects, each object representing a layer in the neural network.
    The JSON must be a string and must be parsable by JSON.parse. JUST RETURN THE JSON STRING, DO NOT USE THE JSON INDICATOR \`\`\`json\`\`\` AT THE BEGINING!!!
    DO NOT TALK ABOUT THE ARCHITECTURE, JUST RETURN THE JSON STRING!!!
    RETURN A VALID JSON PARSABLE STRING!!! DO NOT USE THE JSON INDICATOR \`\`\`json\`\`\ AT THE BEGINING!!!
`;

export const userPrompt = (dataset: string) => {
  return `Get suggestions for the ${dataset} dataset`;
};
