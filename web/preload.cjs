const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Training operations
  startTraining: (config) => ipcRenderer.invoke('start-training', config),
  pauseTraining: () => ipcRenderer.invoke('pause-training'),
  resumeTraining: () => ipcRenderer.invoke('resume-training'),
  stopTraining: () => ipcRenderer.invoke('stop-training'),
  
  // Generation operations
  generateNotebook: (config) => ipcRenderer.invoke('generate-notebook', config),
  
  // Health check
  checkPythonHealth: () => ipcRenderer.invoke('check-python-health'),
  
  // Event listeners for training progress
  onTrainingProgress: (callback) => {
    ipcRenderer.on('training-progress', (event, data) => callback(data));
  },
  
  onTrainingPaused: (callback) => {
    ipcRenderer.on('training-paused', (event, data) => callback(data));
  },
  
  onTrainingResumed: (callback) => {
    ipcRenderer.on('training-resumed', (event, data) => callback(data));
  },
  
  onTrainingStopped: (callback) => {
    ipcRenderer.on('training-stopped', (event, data) => callback(data));
  },
  
  // Remove event listeners
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  }
});

// Expose a flag to indicate we're running in Electron
contextBridge.exposeInMainWorld('isElectron', true);
