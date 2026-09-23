import { useMutation } from "@tanstack/react-query";
import { Config, TransformerConfig } from "~/types/index";
import { API_CONFIG } from "~/util/config";

// Electron API functions
const downloadFile = async (config: Config): Promise<Blob> => {
  const response = await fetch(API_CONFIG.getApiUrl("/generate"), {
    method: "POST",
    body: JSON.stringify(config),
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
  });

  if (!response.ok) {
    throw new Error(
      `Download failed: ${response.status} ${response.statusText}`,
    );
  }
};

const startTraining = async (config: Config) => {
  const response = await fetch(API_CONFIG.getApiUrl("/train"), {
    method: "POST",
    body: JSON.stringify(config),
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
  });

  if (!response.ok) {
    throw new Error(
      `Training failed: ${response.status} ${response.statusText}`,
    );
  }
};

const startTransformerTraining = async (config: TransformerConfig) => {
  const response = await fetch(
    API_CONFIG.getApiUrl("/transformertrain"),
    {
      method: "POST",
      body: JSON.stringify(config),
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
      },
    },
  );

  if (!response.ok) {
    throw new Error(
      `Transformer training failed: ${response.status} ${response.statusText}`,
    );
  }
};

const transformerTest = async (params: {
  temperature: number;
  prompt: string;
}) => {
  const response = await fetch(
    API_CONFIG.getApiUrl("/transformertest"),
    {
      method: "POST",
      body: JSON.stringify(params),
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
      },
    },
  );

  if (!response.ok) {
    throw new Error(
      `Transformer test failed: ${response.status} ${response.statusText}`,
    );
  }
};

const checkServerHealth = async () => {
  const response = await fetch(API_CONFIG.getApiUrl("/health"), {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",
    },
  });

  if (!response.ok) {
    throw new Error(
      `Health check failed: ${response.status} ${response.statusText}`,
    );
  }
};

// hooks
export const useDownloadFile = () => {
  return useMutation({
    mutationFn: downloadFile,
    onSuccess: (blob) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = "generated_notebook.ipynb";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    },
    onError: (error) => {
      console.error("Error downloading file:", error);
    },
  });
};

export const useStartTraining = () => {
  return useMutation({
    mutationFn: startTraining,
    onSuccess: (data) => {
      console.log("Training completed:", data);
    },
    onError: (error) => {
      console.error("Training error:", error);
    },
  });
};

export const useStartTransformerTraining = () => {
  return useMutation({
    mutationFn: startTransformerTraining,
    onSuccess: (data) => {
      console.log("Transformer training completed:", data);
    },
    onError: (error) => {
      console.error("Transformer training error:", error);
    },
  });
};

export const useTransformerTest = () => {
  return useMutation({
    mutationFn: transformerTest,
  });
};

export const useServerHealth = () => {
  return useMutation({
    mutationFn: checkServerHealth,
  });
};

// Raw API functions for backward compatibility
export {
  downloadFile as downloadFileApi,
  startTraining as startTrainingApi,
  startTransformerTraining as startTransformerTrainingApi,
  transformerTest as transformerTestApi,
  checkServerHealth as checkServerHealthApi,
};
