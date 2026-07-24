// Preload script for The Fixer's Electron wrapper.
//
// The wrapped app is a plain Flask + static JS single-page app that doesn't
// expect any Electron-specific APIs - it just needs to run inside a
// BrowserWindow pointed at http://localhost:8090/. Nothing from the renderer
// currently needs a bridge into Node/Electron, so this file intentionally
// exposes nothing. It exists (rather than being omitted) so contextIsolation
// can stay on and nodeIntegration stays off, and so there's an obvious place
// to add a contextBridge API later if the shell ever needs one (e.g. native
// "Save As" dialogs, drag-and-drop from Finder, etc.).

// const { contextBridge } = require('electron');
// contextBridge.exposeInMainWorld('thefixer', { ... });
