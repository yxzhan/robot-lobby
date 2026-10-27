// py_trees behaviour trees in the Behavior panel: Python run by Pyodide
// (CPython in WASM, public/vendor/pyodide, loaded on first use: some 13 MB)
// with py_trees 2.6 (its wheel there too) and the lobby module
// (public/py/lobby.py: GoTo, Move, Approach, Follow, Stop, Wait, robots()…,
// whose leaves bt.js runs). A script defines `root` (a behaviour) or `tree`
// (a BehaviourTree); PyRunner ticks it every TICK_MS of wall time while the
// simulation runs, as bt.js's Runner does a text tree.

import { FAILURE, IDLE, RUNNING, SUCCESS, TICK_MS, context, leaf } from './bt.js';

const VENDOR = new URL('../vendor/pyodide/', import.meta.url);
const WHEEL = 'py_trees-2.6.0-py2.py3-none-any.whl';
const LOBBY = new URL('../py/lobby.py', import.meta.url);

let loading = null;
let world = null;  // {app, goals}: what the lobby module's leaves drive

// Pyodide with py_trees and lobby (once; later calls get the same).
export function loadPython(onProgress) {
  loading ??= (async () => {
    onProgress?.('loading Python (Pyodide)…');
    const { loadPyodide } = await import(new URL('pyodide.mjs', VENDOR).href);
    const py = await loadPyodide({ indexURL: VENDOR.href });
    onProgress?.('loading py_trees…');
    const [wheel, lobby] = await Promise.all([
      fetch(new URL(WHEEL, VENDOR)).then((r) => { if (!r.ok) throw new Error(`${WHEEL}: ${r.status}`); return r.arrayBuffer(); }),
      fetch(LOBBY).then((r) => r.text()),
    ]);
    py.unpackArchive(new Uint8Array(wheel), 'wheel');
    py.FS.writeFile('/home/pyodide/lobby.py', lobby);
    const ctx = () => {
      if (!world) throw new Error('no world');
      return context(world.app, world.goals);
    };
    const plain = (params) => {
      const p = params?.toJs?.({ dict_converter: Object.fromEntries }) ?? params ?? {};
      params?.destroy?.();
      return p;
    };
    py.registerJsModule('lobby_js', {
      check: (type, params) => { leaf(type, plain(params)); },
      leaf: (type, params) => {
        const l = leaf(type, plain(params));
        return { tick: () => l.tick(ctx()), halt: () => { if (world) l.halt(ctx()); }, get info() { return l.info; } };
      },
      robots: () => (world ? world.goals.list() : []),
      pose: (name) => (world ? context(world.app, world.goals).thing(name)?.pose ?? null : null),
      time: () => world?.app.data?.time ?? 0,
    });
    py.runPython(`
import sys, types
sys.path.insert(0, '/home/pyodide')
sys.modules.setdefault('pydot', types.ModuleType('pydot'))  # only py_trees.display's dot export needs it
import py_trees, lobby
`);
    py.lobby = py.pyimport('lobby');
    py.pyTrees = py.pyimport('py_trees');
    return py;
  })();
  loading.catch(() => { loading = null; });  // may try again
  return loading;
}

// A syntax error of a Python script: {line, column, message}, or null; null
// too while Python is not loaded.
export function checkPython(py, source) {
  if (!py) return null;
  const r = py.lobby._check(source);
  if (!r) return null;
  const [line, column, message] = r.toJs();
  r.destroy();
  return { line, column, message };
}

// The line of `<tree>` (the script) an error is at, the last one in its traceback.
const errorLine = (msg) => {
  const lines = [...String(msg).matchAll(/File "<tree>", line (\d+)/g)];
  return lines.length ? Number(lines.at(-1)[1]) : null;
};
// A Python error's last line ("NameError: name 'x' is not defined").
const lastLine = (msg) => String(msg).trim().split('\n').at(-1).replace(/^pyodide\.ffi\.JsException: (\w*Error: )?/, '');

const STATUS = { running: RUNNING, success: SUCCESS, failure: FAILURE, invalid: IDLE };

export class PyRunner {
  // Runs `source` with Pyodide `py` on app / goals; `onTick(runner)` after each tick.
  constructor(py, source, { app, goals, onTick }) {
    this.py = py;
    this.app = app;
    this.goals = goals;
    this.onTick = onTick;
    this.status = RUNNING;
    this.error = null;
    this.errorLine = null;
    this.start = null;
    world = { app, goals };
    const scope = py.globals.get('dict')();
    try {
      py.runPython(source, { globals: scope, filename: '<tree>' });
      const tree = scope.get('tree'), root = scope.get('root');
      if (tree) this.tree = tree;
      else if (root) this.tree = py.pyTrees.trees.BehaviourTree(root);
      else throw new Error('the script defines neither root (a behaviour) nor tree (a BehaviourTree)');
      if (typeof this.tree.tick !== 'function') throw new Error('tree is not a BehaviourTree');
      this.tree.setup();
    } catch (err) {
      this.fail(err);
      this.onTick?.(this);
      return;
    } finally {
      scope.destroy();
    }
    this.timer = setInterval(() => this.step(), TICK_MS);
    this.step();
  }

  fail(err) {
    const msg = err.message ?? String(err);
    this.error = lastLine(msg);
    this.errorLine = err.line ?? errorLine(msg);
    this.status = FAILURE;
    clearInterval(this.timer);
  }

  step() {
    if (this.status !== RUNNING || this.app.paused || !this.app.data) return;
    try {
      world = { app: this.app, goals: this.goals };
      const time = this.app.data.time;
      this.start ??= time;
      if (time < this.start) this.start = time;
      this.time = time - this.start;
      this.tree.tick();
      const s = STATUS[this.tree.root.status.value.toLowerCase()];
      if (s !== RUNNING) this.finish(s);
    } catch (err) {
      this.halt();
      this.fail(err);
    }
    this.onTick?.(this);
  }

  finish(status) {
    clearInterval(this.timer);
    this.status = status;
  }

  halt() {
    try { this.tree?.interrupt(); } catch { /* the world is gone */ }
  }

  // Stops the tree: its running behaviours are interrupted (their goals go).
  stop(status = IDLE) {
    if (this.status !== RUNNING) return;
    this.halt();
    this.finish(status);
    this.onTick?.(this);
  }

  // The tree view's rows: [{depth, text, title, status, info}].
  rows() {
    if (!this.tree) return [];
    const rows = this.py.lobby._rows(this.tree.root);
    const out = rows.toJs().map(([depth, cls, name, status, info]) => ({
      depth, text: name === cls ? cls : name, title: cls, status: STATUS[status] ?? IDLE, info: status === 'invalid' ? '' : info,
    }));
    rows.destroy();
    return out;
  }
}
