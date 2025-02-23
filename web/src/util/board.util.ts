import { Config, UILayer } from "~/types";

export const getConfig = (
  input: string,
  blocks: UILayer[],
  loss: string,
  optimizer: string,
  learningRate: number,
  epoch: number,
  batch_size: number,
) => {
  const layers = [];
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i]!;

    const currentNeuron = block.neurons;
    const nextNeuron = blocks[i + 1]?.neurons || 1; // Default to 1 if no next block, could change based on the dataset
    layers.push({
      kind: block.label,
      args: block.otherParam
        ? [currentNeuron, nextNeuron, block.otherParam]
        : [currentNeuron, nextNeuron],
    });

    if (block.activationFunction) {
      layers.push({
        kind: block.activationFunction,
      });
    }
  }

  const config = {
    input,
    layers,
    loss,
    optimizer: { kind: optimizer, lr: learningRate },
    epoch,
    batch_size,
    learning_rate: learningRate,
  };

  return config;
};

export const downloadFile = async (config: any) => {
  await fetch("http://127.0.0.1:5000/generate", {
    method: "POST",
    body: JSON.stringify(config),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.blob();
    })
    .then((blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = "generated_notebook.ipynb";

      document.body.appendChild(a);
      a.click();

      // clean up
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    })
    .catch((error) => {
      console.error("Error downloading file:", error);
    });
};

export const startTraining = async (config: Config) => {
  return await fetch("http://127.0.0.1:5000/train", {
    method: "POST",
    body: JSON.stringify(config),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((res) => {
      return res.json();
    })
    .then((data) => {
      console.log(data);
      return data;
    });
};

export const getArchitectureSuggestion = async (dataset: string) => {
  return await fetch("/api/get-suggestions", {
    method: "POST",
    body: JSON.stringify({ dataset }),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((res) => {
      return res.json();
    })
    .then((data) => {
      return data;
    });
};
