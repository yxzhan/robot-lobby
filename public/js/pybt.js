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
      body: (name) => bodyState(name),
      upright: (name) => {
        const r = robotOf(name);
        return r ? world.app.data.xmat[9 * r.bodyOffset + 8] : null;
      },
      skill: (name, skill, duration) => {
        const r = robotOf(name);
        if (!r) throw new Error(`no robot ${name}`);
        const c = r.use?.policy && r.policy;
        if (!c?.skillMenu) throw new Error(`${name} has no skills (policy: ${r.use?.policy || 'off'})`);
        c.setSkill(skill, duration ?? undefined);
      },
      set_goal: (name, x, y, yaw) => { world.goals.set(name, { x, y, yaw }); },
      stop: (name) => { world.goals.clear(name); },
      reset: () => window.lobby.reset(),
      place: (name, x, y, yaw, spawn) => {
        const r = robotOf(name);
        if (!r) throw new Error(`no robot ${name}`);
        // Its height and tilt kept, turned about z to `yaw`.
        const { data } = world.app, b = r.bodyOffset, [w, qx, qy, qz] = data.xquat.subarray(4 * b, 4 * b + 4);
        const h = (yaw - Math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))) / 2, c = Math.cos(h), s = Math.sin(h);
        const quat = [c * w - s * qz, c * qx - s * qy, c * qy + s * qx, c * qz + s * w];  // (cos h, 0, 0, sin h) * q
        if (!world.app.placeFree(r, [x, y, data.xpos[3 * b + 2]], quat, !!spawn)) throw new Error(`${name} has a fixed base`);
      },
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

// A spawned thing by name (r2) or label (window.lobby's robots).
const robotOf = (w) => (world ? window.lobby.robots().find((r) => r.prefix.slice(0, -1) === w || r.label === w) ?? null : null);

// {x, y, z, yaw, up} of the world body `name` ('r1/ball', or 'ball': the
// first body of that name in any model), or null; up: the z of its z axis.
// Names from the model (in network mode the server's, the same call).
let bodies = { model: null, ids: new Map() };
function bodyState(name) {
  const app = world?.app, m = app?.model, d = app?.data;
  if (!m || !d) return null;
  if (bodies.model !== m) bodies = { model: m, ids: new Map() };  // (a rebuilt world: other numbers)
  let id = bodies.ids.get(name);
  if (id === undefined) {
    id = -1;
    for (let i = 1; i < m.nbody; i++) {
      const n = app.mujoco.mj_id2name(m, app.mujoco.mjtObj.mjOBJ_BODY.value, i) ?? '';
      if (n === name || n.endsWith(`/${name}`)) { id = i; break; }
    }
    bodies.ids.set(name, id);
  }
  if (id < 0) return null;
  const R = d.xmat.subarray(9 * id, 9 * id + 9);
  return { x: d.xpos[3 * id], y: d.xpos[3 * id + 1], z: d.xpos[3 * id + 2], yaw: Math.atan2(R[3], R[0]), up: R[8] };
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

  // Stops the behaviours running (terminate(INVALID): their goals go).
  // (BehaviourTree.interrupt() only ends its own tick_tock loop.)
  halt() {
    try { this.tree?.root.stop(this.py.pyTrees.common.Status.INVALID); } catch (err) { console.warn('behaviour tree stop:', err); }
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
