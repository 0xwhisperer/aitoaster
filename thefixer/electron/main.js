// The Fixer - Electron main process.
//
// Responsibilities:
//   1. On launch, spawn the Flask backend (./run.sh in the project root,
//      which itself execs venv/bin/python3 -m app.server) as a child
//      process, unless something is already answering on port 8090.
//   2. Poll http://localhost:8090/ until it responds (or time out), showing
//      a small loading window in the meantime.
//   3. Open the real BrowserWindow pointed at http://localhost:8090/.
//   4. On quit, kill the backend child process tree so no orphaned Python
//      process is left running.
//
// Packaging note (see ../electron/README.md for the full story): this
// currently assumes a working venv/ already exists alongside the project
// (thefixer/venv), i.e. Option (a) from the task brief. It does NOT bundle
// Python, torch, onnxruntime, ffmpeg, etc. for end users who don't already
// have this dev environment set up.

const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

const BACKEND_HOST = '127.0.0.1';
const BACKEND_PORT = 8090;
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}/`;

// Repo layout: <repo>/thefixer/electron/main.js -> project root is one level up.
const PROJECT_ROOT = path.resolve(__dirname, '..');
const RUN_SCRIPT = path.join(PROJECT_ROOT, 'run.sh');

const POLL_INTERVAL_MS = 500;
const STARTUP_TIMEOUT_MS = 90_000; // torch/onnxruntime import + model load can be slow on first boot

const isDev = process.env.ELECTRON_DEV === '1';

/** @type {import('child_process').ChildProcess | null} */
let backendProcess = null;
let backendOwnedByUs = false; // only true if *we* spawned it (vs. it already being up)
let mainWindow = null;
let loadingWindow = null;
let quitting = false;

function checkBackendAlive() {
  return new Promise((resolve) => {
    const req = http.get(BACKEND_URL, { timeout: 2000 }, (res) => {
      res.resume();
      resolve(true);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

function waitForBackend(timeoutMs) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    (async function poll() {
      if (await checkBackendAlive()) {
        resolve();
        return;
      }
      if (Date.now() - start > timeoutMs) {
        reject(new Error(`Backend did not respond on ${BACKEND_URL} within ${timeoutMs}ms`));
        return;
      }
      setTimeout(poll, POLL_INTERVAL_MS);
    })();
  });
}

function spawnBackend() {
  backendProcess = spawn(RUN_SCRIPT, [], {
    cwd: PROJECT_ROOT,
    // Detach so the child gets its own process group on POSIX - lets us
    // kill the whole group (run.sh + the python3 it execs) on quit, rather
    // than just the immediate shell.
    detached: process.platform !== 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
    env: process.env,
  });
  backendOwnedByUs = true;

  backendProcess.stdout.on('data', (chunk) => {
    process.stdout.write(`[thefixer-backend] ${chunk}`);
  });
  backendProcess.stderr.on('data', (chunk) => {
    process.stderr.write(`[thefixer-backend] ${chunk}`);
  });
  backendProcess.on('exit', (code, signal) => {
    console.log(`[thefixer-backend] exited (code=${code}, signal=${signal})`);
    if (!quitting) {
      // Backend died unexpectedly while the app was still meant to be running.
      backendProcess = null;
    }
  });
  backendProcess.on('error', (err) => {
    console.error('[thefixer-backend] failed to spawn:', err);
  });
}

function killBackend() {
  if (!backendProcess || !backendOwnedByUs) return;
  quitting = true;
  try {
    if (process.platform === 'win32') {
      // No process groups on Windows; taskkill /T walks the child tree
      // (run.sh's cmd equivalent + the python3 it launched).
      spawn('taskkill', ['/pid', String(backendProcess.pid), '/T', '/F']);
    } else {
      // Negative pid = signal the whole process group (run.sh + exec'd python3).
      process.kill(-backendProcess.pid, 'SIGTERM');
      // Belt-and-suspenders: escalate to SIGKILL if it's still alive shortly after.
      setTimeout(() => {
        try {
          process.kill(-backendProcess.pid, 'SIGKILL');
        } catch (_) {
          // already gone
        }
      }, 3000);
    }
  } catch (err) {
    console.error('Error killing backend process:', err);
  }
}

function createLoadingWindow() {
  loadingWindow = new BrowserWindow({
    width: 420,
    height: 280,
    resizable: false,
    frame: false,
    center: true,
    show: true,
    backgroundColor: '#111318',
    webPreferences: {
      contextIsolation: true,
    },
  });
  loadingWindow.loadFile(path.join(__dirname, 'loading.html'));
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    show: false,
    backgroundColor: '#111318',
    title: 'The Fixer',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      devTools: true,
    },
  });

  mainWindow.once('ready-to-show', () => {
    if (loadingWindow && !loadingWindow.isDestroyed()) {
      loadingWindow.close();
      loadingWindow = null;
    }
    mainWindow.show();
    if (isDev) {
      mainWindow.webContents.openDevTools({ mode: 'detach' });
    }
  });

  mainWindow.loadURL(BACKEND_URL);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

async function boot() {
  createLoadingWindow();

  const alreadyRunning = await checkBackendAlive();
  if (!alreadyRunning) {
    console.log(`[thefixer] No backend detected on ${BACKEND_URL}, spawning ${RUN_SCRIPT}`);
    spawnBackend();
  } else {
    console.log(`[thefixer] Backend already reachable on ${BACKEND_URL}, not spawning a new one`);
    backendOwnedByUs = false;
  }

  try {
    await waitForBackend(STARTUP_TIMEOUT_MS);
  } catch (err) {
    console.error(err.message);
    if (loadingWindow && !loadingWindow.isDestroyed()) {
      loadingWindow.close();
    }
    dialog.showErrorBox(
      'The Fixer failed to start',
      `The backend server did not become reachable at ${BACKEND_URL} within ` +
        `${STARTUP_TIMEOUT_MS / 1000}s.\n\n` +
        `Check that:\n` +
        `  - ${RUN_SCRIPT} exists and is executable\n` +
        `  - thefixer/venv/ has dependencies installed (see requirements.txt)\n` +
        `  - ffmpeg is installed and on PATH\n\n` +
        `Details: ${err.message}`
    );
    app.quit();
    return;
  }

  createMainWindow();
}

app.whenReady().then(boot);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    boot();
  }
});

app.on('before-quit', () => {
  killBackend();
});

// Safety net: also try on these signals/events in case before-quit doesn't
// fire (e.g. process killed externally while dev-testing).
process.on('exit', killBackend);
process.on('SIGINT', () => {
  killBackend();
  process.exit(0);
});
process.on('SIGTERM', () => {
  killBackend();
  process.exit(0);
});
