export interface ElectronAPI {
  // Training operations
  startTraining: (config: any) => Promise<any>;
  pauseTraining: () => Promise<any>;
  resumeTraining: () => Promise<any>;
  stopTraining: () => Promise<any>;

  // Generation operations
  generateNotebook: (config: any) => Promise<any>;

  // Health check
  checkPythonHealth: () => Promise<{
    status: string;
    message?: string;
    error?: string;
  }>;

  // Event listeners
  onTrainingProgress: (callback: (data: any) => void) => void;
  onTrainingPaused: (callback: (data: any) => void) => void;
  onTrainingResumed: (callback: (data: any) => void) => void;
  onTrainingStopped: (callback: (data: any) => void) => void;

  // Cleanup
  removeAllListeners: (channel: string) => void;
}

declare global {
  interface Window {
    electronAPI: ElectronAPI;
    isElectron: boolean;
  }
}

export {};
