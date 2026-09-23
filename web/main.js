const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const isDev = process.env.NODE_ENV === 'development';
const { PythonShell } = require('python-shell');

let mainWindow;

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: 'hiddenInset',
    icon: path.join(__dirname, 'public/favicon.png')
  });

  // Maximize the window
  mainWindow.maximize();

  // Load the app
  if (isDev) {
    mainWindow.loadURL('http://localhost:3000');
    // Open the DevTools in development
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, 'out/index.html'));
  }
}

// This method will be called when Electron has finished initialization
app.whenReady().then(createWindow);

// Quit when all windows are closed, except on macOS
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Python execution handlers
ipcMain.handle('start-training', async (event, config) => {
  try {
    const scriptPath = path.join(__dirname, 'electron', 'python-scripts');
    const options = {
      mode: 'json',
      pythonPath: 'python3',
      scriptPath: scriptPath,
      args: [JSON.stringify(config)]
    };

    return new Promise((resolve, reject) => {
      const pyshell = new PythonShell('training_script.py', options);
      
      let results = [];
      
      pyshell.on('message', (data) => {
        console.log('Python output:', data);
        
        // Only send progress updates (messages with epoch info) to renderer
        // Skip status messages like "completed", "starting", etc.
        if (data.epoch && data.total_epochs) {
          mainWindow.webContents.send('training-progress', data);
        }
        
        results.push(data);
      });

      pyshell.end((err, code, signal) => {
        if (err) {
          console.error('Python script error:', err);
          reject(err);
        } else {
          console.log('Python script completed with code:', code);
          
          // Find the completion message and extract final results
          const completionMessage = results.find(msg => msg.status === 'completed');
          const finalResults = completionMessage ? completionMessage.results : null;
          
          resolve({ results: finalResults, code, signal });
        }
      });
    });
  } catch (error) {
    console.error('Training error:', error);
    throw error;
  }
});

ipcMain.handle('generate-notebook', async (event, config) => {
  try {
    const scriptPath = path.join(__dirname, 'electron', 'python-scripts');
    const options = {
      mode: 'json',
      pythonPath: 'python3',
      scriptPath: scriptPath,
      args: [JSON.stringify(config)]
    };

    return new Promise((resolve, reject) => {
      PythonShell.run('generate_notebook.py', options, (err, results) => {
        if (err) {
          console.error('Notebook generation error:', err);
          reject(err);
        } else {
          console.log('Notebook generated successfully');
          resolve(results);
        }
      });
    });
  } catch (error) {
    console.error('Generate notebook error:', error);
    throw error;
  }
});

ipcMain.handle('check-python-health', async () => {
  try {
    const scriptPath = path.join(__dirname, 'electron', 'python-scripts');
    const options = {
      mode: 'json',
      pythonPath: 'python3',
      scriptPath: scriptPath
    };

    return new Promise((resolve, reject) => {
      PythonShell.run('health_check.py', options, (err, results) => {
        if (err) {
          console.error('Health check error:', err);
          resolve({ status: 'offline', error: err.message });
        } else {
          console.log('Health check results:', results);
          const result = results && results.length > 0 ? results[0] : { status: 'online', message: 'Python backend is running' };
          resolve(result);
        }
      });
    });
  } catch (error) {
    console.error('Health check exception:', error);
    return { status: 'offline', error: error.message };
  }
});

// Handle training control
ipcMain.handle('pause-training', async () => {
  // Send signal to pause training (implementation depends on Python script structure)
  mainWindow.webContents.send('training-paused');
  return { message: 'Training paused' };
});

ipcMain.handle('resume-training', async () => {
  // Send signal to resume training
  mainWindow.webContents.send('training-resumed');
  return { message: 'Training resumed' };
});

ipcMain.handle('stop-training', async () => {
  // Send signal to stop training
  mainWindow.webContents.send('training-stopped');
  return { message: 'Training stopped' };
});
