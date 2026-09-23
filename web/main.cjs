const { app, BrowserWindow, ipcMain, protocol } = require('electron');
const path = require('path');
const { readFile } = require('fs/promises');
const isDev = !app.isPackaged;
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
      webSecurity: false, // Disable web security for local files
      preload: path.join(__dirname, 'preload.cjs')
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
    // Load the index.html file from the out directory
    const indexPath = path.join(__dirname, 'out', 'index.html');
    mainWindow.loadFile(indexPath);
  }
  
  // Open DevTools for debugging (uncomment when needed)
  // if (!isDev) {
  //   mainWindow.webContents.openDevTools();
  // }
}

// This method will be called when Electron has finished initialization
app.whenReady().then(() => {
  // Register a custom protocol to serve static files
  if (!isDev) {
    protocol.registerFileProtocol('app', (request, callback) => {
      const url = request.url.substr(6); // Remove 'app://' prefix
      const filePath = path.join(__dirname, 'out', url);
      callback(filePath);
    });
  }
  
  createWindow();
});

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
function getPythonScriptsPath() {
  if (isDev) {
    return path.join(__dirname, 'electron', 'python-scripts');
  }
  return path.join(process.resourcesPath, 'app.asar.unpacked', 'electron', 'python-scripts');
}
ipcMain.handle('start-training', async (event, config) => {
  try {
    const scriptPath = getPythonScriptsPath();
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
    const scriptPath = getPythonScriptsPath();
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
    const scriptPath = getPythonScriptsPath();
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
