// API Configuration
export const API_CONFIG = {
  // Use environment variable if available, otherwise fallback to ngrok URL for development
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000",

  // Common headers for API requests
  getHeaders: () => ({
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true",
  }),

  // Helper function to get full API URL
  getApiUrl: (endpoint: string) => {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";
    return `${baseUrl}${endpoint}`;
  },

  // Specific endpoint URLs
  endpoints: {
    generate: "/generate",
    train: "/train",
    trainStream: "/train-stream",
    transformerTrain: "/transformertrain",
    transformerTest: "/transformertest",
    health: "/health",
  },
};

// Socket Configuration
export const SOCKET_CONFIG = {
  URL:
    process.env.NEXT_PUBLIC_SOCKET_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:5000",
  options: {
    transports: ["websocket"],
    autoConnect: true,
  },
};
