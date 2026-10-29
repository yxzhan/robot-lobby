// py_trees behaviour trees in the Behavior panel: Python run by Pyodide
// (CPython in WASM, public/vendor/pyodide, loaded on first use: some 13 MB)
// with py_trees 2.6 (its wheel there too) and the lobby module
// (public/py/lobby.py: GoTo, Move, Approach, Follow, Stop, Wait, robots()…,
// whose leaves bt.js runs). A script defines `root` (a behaviour) or `tree`
// (a BehaviourTree); PyRunner ticks it every TICK_SIM of simulated time
// while the simulation runs, as bt.js's Runner does a text tree.

import { FAILURE, IDLE, POLL_MS, RUNNING, SUCCESS, context, due, leaf } from './bt.js';

const VENDOR = new URL('../vendor/pyodide/', import.meta.url);
const WHEEL = 'py_trees-2.6.0-py2.py3-none-any.whl';
const LOBBY = new URL('../py/lobby.py', import.meta.url);

let loading = null;
let world = null;  // {app, goals, confirm}: what the lobby module's leaves drive

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
      reset: () => { ask('The behaviour tree wants to reset the scene.'); window.lobby.reset(); },
      search: (query, category) => window.lobby.search(query, category || null),
      remove: (names) => {
        const list = String(names).split(',').map((n) => robotOf(n) ?? fail(`no robot ${n}`));
        if (!list.length) return;
        ask(`The behaviour tree wants to remove ${list.map((r) => r.prefix.slice(0, -1)).join(', ')}.`);
        window.lobby.remove(list).catch((err) => console.warn('remove:', err));
      },
      clear: () => {
        const list = window.lobby.robots();
        if (!list.length) return;
        ask(`The behaviour tree wants to remove all ${list.length} spawned models.`);
        window.lobby.remove(list).catch((err) => console.warn('clear:', err));
      },
      pause: () => { window.lobby.setPaused(true).catch((err) => console.warn('pause:', err)); },
      physics: (changes) => {
        const c = plain(changes);
        if (Object.keys(c).length) window.lobby.setPhysics(c).catch((err) => console.warn('physics:', err));
        return structuredClone(world.app.physics);
      },
      set_controller: (name, kind, on, policy) => {
        const r = robotOf(name) ?? fail(`no robot ${name}`);
        if (!['policy', 'base', 'ik'].includes(kind)) fail(`no controller ${kind} (policy, base, ik)`);
        window.lobby.setController(r, kind, on, policy || undefined).catch((err) => console.warn('set_controller:', err));
      },
      inspect: (name, only) => {
        const r = robotOf(name) ?? fail(`no robot ${name}`);
        return JSON.stringify(window.lobby.describe(r, only || undefined));  // (as JSON: nested nulls would be jsnull)
      },
      controllers: (name) => {
        const r = robotOf(name);
        if (!r) return null;
        const c = r.use?.policy && r.policy, menu = c?.skillMenu;
        return { policy: r.use?.policy || undefined, base: !!r.use?.base, ik: !!r.use?.ik, skills: menu ? [...menu.manual] : [] };
      },
      place: (name, x, y, yaw, spawn, pitch, z) => {
        const r = robotOf(name);
        if (!r) throw new Error(`no robot ${name}`);
        const { data } = world.app, b = r.bodyOffset;
        let quat;
        if (pitch == null) {  // its tilt kept, turned about z to `yaw`
          const [w, qx, qy, qz] = data.xquat.subarray(4 * b, 4 * b + 4);
          const h = (yaw - Math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))) / 2, c = Math.cos(h), s = Math.sin(h);
          quat = [c * w - s * qz, c * qx - s * qy, c * qy + s * qx, c * qz + s * w];  // (cos h, 0, 0, sin h) * q
        } else {  // yaw, then pitched about its own y axis (+90°: on its face, −90°: on its back)
          const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2), cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
          quat = [cy * cp, -sy * sp, cy * sp, sy * cp];
        }
        if (!world.app.placeFree(r, [x, y, z ?? data.xpos[3 * b + 2]], quat, !!spawn)) throw new Error(`${name} has a fixed base`);
      },
      place_body: (name, x, y, z) => {
        const at = bodyState(name);
        if (!at) throw new Error(`no body ${name}`);
        if (!world.app.placeBody(at.name, [x, y, z ?? at.z])) throw new Error(`${name} is not on a free joint of its own`);
      },
      spawn: (model, x, y, yaw, z) => {
        const asset = window.lobby.assets().find((a) => a.rel === model || a.name === model);
        if (!asset) throw new Error(`no model ${model} in the library`);
        window.lobby.spawn(asset.rel, [x, y, z ?? 0], yaw).catch((err) => console.warn('spawn:', err));  // (it takes a moment)
      },
      set_policy: (name, id) => {
        const r = robotOf(name);
        if (!r) throw new Error(`no robot ${name}`);
        window.lobby.setController(r, 'policy', !!id, id || undefined).catch((err) => console.warn('set_policy:', err));
      },
      policy: (name) => robotOf(name)?.use?.policy || null,
      active_skill: (name) => {
        const r = robotOf(name), c = r?.use?.policy && r.policy;
        return c ? (c.activeSkill ?? c.skills?.default ?? null) : null;
      },
      // IK chains (ik.js; remote.js in network mode): as JSON, poses in the world.
      ik_chains: (name) => {
        const r = robotOf(name) ?? fail(`no robot ${name}`), ik = ikOf(r, false), d = world.app.data;
        return JSON.stringify((ik?.chains ?? []).map((c) => ({
          name: c.cfg.name, tip: c.cfg.tip, gripper: !!c.gripper?.length, error: c.error || undefined,
          target: poseOut(!c.error && ik.targetPose(c, d)), tip_pose: poseOut(!c.error && tipPose(ik, c, d)),
          ...(c.err && { err: { pos: c.err[0], rot: c.err[1] } }),
        })));
      },
      set_ik_target: (name, chain, pos, quat) => {
        const r = robotOf(name) ?? fail(`no robot ${name}`), ik = ikOf(r), c = chainOf(r, ik, chain), d = world.app.data;
        const p = plain(pos), q = quat == null ? null : plain(quat);
        if (p.length !== 3 || p.some((v) => !Number.isFinite(v))) fail('pos: [x, y, z]');
        const rot = q ? quatToMat(q) : (ik.targetPose(c, d) ?? tipPose(ik, c, d)).rot;
        ik.setTargetPose(c, d, { pos: p, rot });
      },
      set_gripper: (name, chain, amount) => {
        const r = robotOf(name) ?? fail(`no robot ${name}`), ik = ikOf(r), c = chainOf(r, ik, chain);
        if (!c.gripper?.length) fail(`${c.cfg.name} of ${name} has no gripper`);
        ik.setGripper(c, world.app.data, Math.min(1, Math.max(0, +amount)));
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

const fail = (msg) => { throw new Error(msg); };
// Asks the user first when the tree's starter wants that (the Agent's trees);
// declined: an error, the tree fails.
const ask = (what) => { if (world?.confirm && !world.confirm(what)) fail('the user declined'); };

// A spawned thing by name (r2) or label (window.lobby's robots).
const robotOf = (w) => (world ? window.lobby.robots().find((r) => r.prefix.slice(0, -1) === w || r.label === w) ?? null : null);

// Robot `r`'s IK (its chains), on (else, with `strict`, an error saying so).
function ikOf(r, strict = true) {
  const name = r.prefix.slice(0, -1);
  if (!r.ik) return strict ? fail(`${name} has no IK chains`) : null;
  if (strict && !r.use?.ik) fail(`IK is off for ${name}: lobby.set_controller('${name}', 'ik', True)`);
  return r.ik;
}

// The chain `which` (a name, or its index; null: the only one).
function chainOf(r, ik, which) {
  const chains = ik.chains ?? [], name = r.prefix.slice(0, -1);
  const c = which == null ? (chains.length === 1 ? chains[0] : null)
    : typeof which === 'number' ? chains[which] : chains.find((x) => x.cfg.name === which);
  if (!c) fail(`${name}: no IK chain ${which ?? '(it has several: name one)'} (${chains.map((x) => x.cfg.name).join(', ')})`);
  if (c.error) fail(`${name}'s IK chain ${c.cfg.name}: ${c.error}`);
  return c;
}

// Where a chain's tip is: its TCP site, else its tip body moved by the
// offset (in network mode from the server's site tables and the bodies'
// poses: the page computes no sites).
function tipPose(ik, c, d) {
  if (ik.tipFrame) return ik.tipFrame(d, c);
  const m = world.app.model, s = c.cfg.site ? ik.id('mjOBJ_SITE', c.cfg.site) : -1;
  const b = s >= 0 ? m.site_bodyid[s] : ik.id('mjOBJ_BODY', c.cfg.tip);
  if (b < 0) return null;
  const R = Array.from(d.xmat.subarray(9 * b, 9 * b + 9));
  const o = s >= 0 ? Array.from(m.site_pos.subarray(3 * s, 3 * s + 3)) : c.cfg.offset ?? [0, 0, 0];
  const pos = [0, 1, 2].map((i) => d.xpos[3 * b + i] + R[3 * i] * o[0] + R[3 * i + 1] * o[1] + R[3 * i + 2] * o[2]);
  if (s < 0) return { pos, rot: R };
  const S = quatToMat(Array.from(m.site_quat.subarray(4 * s, 4 * s + 4)));
  return { pos, rot: [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => R[3 * i] * S[j] + R[3 * i + 1] * S[3 + j] + R[3 * i + 2] * S[6 + j])) };
}

const poseOut = (p) => (p ? { pos: p.pos.map((v) => +v.toFixed(5)), quat: matToQuat(p.rot).map((v) => +v.toFixed(5)) } : undefined);

// Row-major 3x3 <-> unit quaternion (w, x, y, z), as MuJoCo has them.
function quatToMat([w, x, y, z]) {
  const n = Math.hypot(w, x, y, z) || 1;
  [w, x, y, z] = [w / n, x / n, y / n, z / n];
  return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
}
function matToQuat(m) {
  const [a, b, c, d, e, f, g, h, i] = m, t = a + e + i;
  let q;
  if (t > 0) { const s = 2 * Math.sqrt(t + 1); q = [s / 4, (h - f) / s, (c - g) / s, (d - b) / s]; }
  else if (a > e && a > i) { const s = 2 * Math.sqrt(1 + a - e - i); q = [(h - f) / s, s / 4, (b + d) / s, (c + g) / s]; }
  else if (e > i) { const s = 2 * Math.sqrt(1 + e - a - i); q = [(c - g) / s, (b + d) / s, s / 4, (f + h) / s]; }
  else { const s = 2 * Math.sqrt(1 + i - a - e); q = [(d - b) / s, (c + g) / s, (f + h) / s, s / 4]; }
  return q[0] < 0 ? q.map((v) => -v) : q;
}

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
  return { name: app.mujoco.mj_id2name(m, app.mujoco.mjtObj.mjOBJ_BODY.value, id), x: d.xpos[3 * id], y: d.xpos[3 * id + 1], z: d.xpos[3 * id + 2],
           yaw: Math.atan2(R[3], R[0]), up: R[8] };
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
  constructor(py, source, { app, goals, confirm, onTick }) {
    this.py = py;
    this.app = app;
    this.goals = goals;
    this.confirm = confirm;
    this.onTick = onTick;
    this.status = RUNNING;
    this.error = null;
    this.errorLine = null;
    this.start = null;
    world = { app, goals, confirm };
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
    this.timer = setInterval(() => this.step(), POLL_MS);
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
    if (this.status !== RUNNING || this.app.paused || !this.app.data || !due(this, this.app.data.time)) return;
    try {
      world = { app: this.app, goals: this.goals, confirm: this.confirm };
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
