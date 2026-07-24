# The Fixer - Electron Desktop Wrapper

Wraps the existing Flask app (`../app/server.py`, served at `http://localhost:8090/`)
in a native desktop shell: launch the app, and it starts the Python backend for
you, waits for it to come up, and opens a window on it. Quit the app and the
backend is killed with it.

This directory is purely additive - nothing under `../app/` or `../static/`
was touched.

## Prerequisites (host machine, dev mode)

- Node.js + npm (tested with Node v22, npm 10)
- The existing `../venv/` must already exist with dependencies installed
  (`cd .. && ./run.sh` once by hand is the easiest way to create it - it
  bootstraps the venv from `requirements.txt` on first run).
- **ffmpeg must be on `PATH`.** The Python backend shells out to a bare
  `ffmpeg` command (see `../app/server.py`, `../app/detector.py`,
  `../app/chain.py`, `../app/linear_fix.py`, `../app/cnn_fix.py`,
  `../app/linear_differentiable.py`, `../app/cnn_differentiable_v2.py` - all
  do `subprocess.run(["ffmpeg", ...])`/`["ffmpeg", ...]` with no absolute
  path). On macOS: `brew install ffmpeg`. This wrapper does **not** bundle a
  static ffmpeg binary - see "Follow-ups" below.

## Commands

```bash
cd electron
npm install          # one-time

npm run dev           # dev mode: devtools auto-open, verbose logging
npm start              # same as dev but without forcing devtools

npm run build:mac      # arm64 + x64 .dmg and .zip -> electron/dist/
npm run build:win      # x64 NSIS .exe installer -> electron/dist/
npm run build:all      # both mac and win
```

`npm run dev` sets `ELECTRON_DEV=1`, which makes `main.js` auto-open DevTools
(detached) on the main window once it loads. There's no bundler/hot-reload
step here because this shell has no build step of its own (`main.js`,
`preload.js`, `loading.html` are loaded directly by Electron) - editing those
files and re-running `npm run dev` is the iteration loop. The wrapped web app
itself (`../static/`) is just fetched fresh from Flask on every window load,
so changes there show up on an Electron reload (Cmd/Ctrl+R) without touching
Electron at all.

## How backend lifecycle works (`main.js`)

1. On `app.whenReady()`, show a small frameless loading window.
2. Probe `http://localhost:8090/` with a plain HTTP GET.
   - If it already answers (e.g. you're also running `./run.sh` by hand for
     backend development), the wrapper does **not** spawn a second backend -
     it just waits for and uses the one that's there, and won't kill it on
     quit (since it didn't start it).
   - Otherwise, spawn `../run.sh` (detached, own process group on POSIX) from
     the project root and poll until `localhost:8090/` responds or 90s pass.
3. Close the loading window, open the real `BrowserWindow` at
   `http://localhost:8090/`.
4. On quit (`before-quit`, plus `SIGINT`/`SIGTERM`/`exit` handlers as a
   safety net): if *this process* spawned the backend, send `SIGTERM` to the
   whole process group (`-pid`, so it reaches both the `run.sh` shell and the
   `python3 -m app.server` it `exec`s into - `run.sh` uses `exec` precisely so
   there's only one PID to manage), with a `SIGKILL` escalation 3s later if
   it's still alive. On Windows, `taskkill /pid <pid> /T /F` walks the same
   process tree.

## Packaging approach for the Python backend: what was actually shipped

**Shipped: Option (a) - spawn from an existing `venv/`, not a frozen binary.**

`main.js` spawns `../run.sh`, which in turn execs
`venv/bin/python3 -m app.server`. This means:

- Works today for anyone who already has this repo cloned with `venv/` set
  up (i.e. you, right now, as a developer).
- **Does not work** as a give-someone-the-.dmg-and-they-double-click-it
  distributable. A clean-machine end user has no Python, no venv, and none of
  torch/onnxruntime/nnAudio/etc. installed, and the packaged `.app`/`.exe`
  from `electron-builder` only bundles the Electron shell itself (`main.js`,
  `preload.js`, `loading.html`) - it does **not** currently bundle
  `../app`, `../venv`, or `../requirements.txt` into `Resources/`. Running
  the built app on a machine without a sibling `thefixer/venv/` will show the
  "failed to start" error dialog after the 90s timeout.

This was a deliberate scope call given the ML dependency footprint (torch +
onnxruntime + nnAudio + torchaudio + torchcodec are individually large and
some have platform-specific wheels), rather than half-shipping a PyInstaller
freeze that silently breaks on missing native deps. See below for what doing
it properly would take.

### What a real PyInstaller-based path would require

To get an actual double-click-and-it-works distributable with no Python
prerequisite:

1. **Freeze the backend with PyInstaller** (`pyinstaller --onedir
   app/server.py` from the project root, or a `.spec` file for finer
   control). Expect to need explicit `--hidden-import`/`--collect-all` flags
   for at least `onnxruntime`, `torch`, `torchaudio`, `torchcodec`,
   `onnx2torch`, and `nnAudio` - these packages do a lot of dynamic/plugin
   style importing that PyInstaller's static analysis doesn't always catch,
   and torch in particular ships large native `.so`/`.dylib` binaries that
   need `--collect-binaries` or manual `datas=[...]` entries in the spec.
   Budget real iteration time here (multiple rebuild-and-test cycles are
   normal) - this is exactly the "substantial work" the task brief flagged.
2. **Verify the frozen binary runs standalone** (`./dist/server/server`
   or similar) with no `venv/` on `PATH` at all, on a clean-ish shell, before
   touching Electron.
3. **Bundle it via `extraResources`** in `electron/package.json`'s `build`
   config, e.g.:
   ```jsonc
   "extraResources": [
     { "from": "../dist/server", "to": "backend" }
   ]
   ```
   then change `main.js` to spawn
   `process.resourcesPath + "/backend/server"` instead of `run.sh` when
   `app.isPackaged` is true (keep the current `run.sh` dev-mode path for
   `npm run dev`).
4. **Bundle a static ffmpeg binary** too (see below) and point the backend at
   it via an env var or absolute path, since a packaged app can't assume
   `ffmpeg` is on the end user's `PATH` either.
5. **Model weights**: check `../models/` - if the scorer/detector loads
   checkpoint files from disk, those need to ship in `extraResources` too, or
   the frozen backend will fail at `Scorer()` init on first request.
6. **Per-platform frozen builds**: PyInstaller freezes for the OS/arch it
   runs on - a macOS arm64 freeze cannot produce a Windows binary. Getting a
   Windows distributable with no Python prerequisite means either running
   PyInstaller on an actual Windows machine/VM/CI runner, or in a Windows
   Docker container - this is a separate build pipeline from the Electron
   Windows cross-build described below, which only cross-compiles the
   *shell*, not the frozen Python backend.
7. **Expect the unpacked output to be large** (500MB-1.5GB range is typical
   once torch + onnxruntime + CUDA-adjacent shared libs are collected, even
   without an actual GPU build) - worth deciding up front whether that's
   acceptable for a shipped installer size, and whether a CPU-only torch
   build trims it meaningfully.

None of this is implemented here; `main.js` has no `app.isPackaged` branch
today, so a built app will always try to spawn `../run.sh` relative to the
`.app`/install directory, which won't exist in a real end-user install.

### Follow-up: bundling a static ffmpeg

Not required for this task, but the natural next step alongside PyInstaller
packaging: drop a static ffmpeg build (e.g. from
[evermeet.cx](https://evermeet.cx/ffmpeg/) for mac,
[gyan.dev](https://www.gyan.dev/ffmpeg/builds/) for Windows) into
`extraResources`, and have the backend prefer an `FFMPEG_PATH` env var (set
by `main.js` when packaged) over the bare `"ffmpeg"` it currently calls. That
requires a small, coordinated change in `../app/` (the files listed above) -
out of scope here since `app/` is explicitly someone else's active work.

## Cross-compiling the Windows build from macOS

`electron-builder --win` was run on this Apple Silicon Mac with Wine Stable
(`brew install --cask wine-stable`, x86_64 binary running under Rosetta 2)
already installed. It worked: `npm run build:win` produced a real,
unsigned NSIS installer (`dist/The Fixer Setup 0.1.0.exe`) and an unpacked
`dist/win-unpacked/The Fixer.exe`, verified as genuine PE32+ Windows
executables (via `file`), not just a successful exit code. No manual Wine
setup beyond having it installed was needed - electron-builder shells out to
`wine`/`makensis` itself. The installer is unsigned (no Windows code-signing
cert configured), same caveat as the unsigned macOS build.
