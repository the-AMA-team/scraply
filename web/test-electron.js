// Test Electron main process
console.log('Starting Electron test...');
const { app, BrowserWindow } = require('electron');

console.log('App imported:', !!app);
console.log('BrowserWindow imported:', !!BrowserWindow);

app.whenReady().then(() => {
  console.log('Electron app is ready!');
  const win = new BrowserWindow({
    width: 800,
    height: 600
  });
  
  win.loadURL('data:text/html,<h1>Hello Electron!</h1>');
});

app.on('window-all-closed', () => {
  app.quit();
});
