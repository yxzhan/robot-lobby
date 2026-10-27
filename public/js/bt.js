// Behaviour trees for a team of robots: one tree drives them all, each
// action naming the robots it moves (the Behavior panel, btpanel.js; from
// the devtools console window.bt). Built on main.js's window.goals, the
// goal markers, so it works in local and network mode alike; the tree runs
// in this page (ticked TICK_MS of wall time apart, on simulation time:
// paused, nothing happens; at 4× speed a Wait is over 4× as fast).
//
// The text form, one node a line, children indented under it, `#` comments:
//
//   Repeat
//     Sequence
//       Parallel
//         GoTo r1 x=1 y=0
//         GoTo r2 x=-1 y=0 yaw=180deg
//       Wait 2
//       Move all forward=0.5
//
// Parameters in their order, or as name=value; angles in rad or with `deg`;
// robots: a name (r3), a label (as in the Stage panel: every robot of it),
// `all` (every robot whose walking policy or base is on) or a list r1,r2
// (no spaces); a target: a name, a label or a part of one (ball). A
// BehaviorTree.CPP XML file (<root BTCPP_format="4"><BehaviorTree>...,
// as Groot2 saves it) is read too: the same nodes as elements, their
// parameters as attributes, Sleep and Timeout in msec, Repeat's num_cycles,
// RetryUntilSuccessful's num_attempts, Parallel's success_count and
// failure_count.
//
// Nodes (status: running, success or failure):
//   Sequence         children in turn; fails with the first that fails
//   Fallback         children in turn until one succeeds (Selector)
//   ReactiveSequence, ReactiveFallback: from the first child every tick (a
//                    child that stops running halts the ones after it)
//   Parallel success failure: all children at once; succeeds once `success`
//                    of them have (default all), fails once `failure` (1)
//   Repeat times     the child `times` times over (-1, the default: forever;
//                    one round a tick), fails when it fails
//   Retry times      the child again after a failure, up to `times` (-1)
//   Timeout seconds  fails (halting the child) if it runs longer
//   Inverter, ForceSuccess, ForceFailure: the child's result turned
//   Wait seconds     simulated time
//   GoTo robot x y yaw   to a pose on the floor (m; yaw: facing the way
//                    there if not given); succeeds when it is reached, fails
//                    if the goal is taken from it (the marker dragged, the
//                    goal cleared, another behaviour)
//   Move robot forward left turn: as GoTo, relative to where each robot is
//                    when it starts (m in its own frame, rad)
//   Approach robot target distance: to `distance` (0.4 m) from `target`
//                    (any spawned thing: a robot, a ball), facing it,
//                    following it while it moves; succeeds when there
//   Follow robot target distance: as Approach, but keeps following (runs
//                    until halted: put it in a Parallel or a Timeout)
//   Stop robot       clears the goals: the robots stop where they are
// Halting a node (a Timeout, a Parallel done, Stop in the panel) clears the
// goals it set.

export const TICK_MS = 100;
export const RUNNING = 'running', SUCCESS = 'success', FAILURE = 'failure', IDLE = 'idle';

const MOVED = 0.1;      // m: Approach/Follow set a goal again once it would move this far
const NEAR = 0.15;      // m over the distance: Approach counts as there (Follow starts again beyond)
const SAME = 1e-6;      // goals this close are the same one (network echoes)

const COMPOSITES = ['Sequence', 'Fallback', 'ReactiveSequence', 'ReactiveFallback', 'Parallel'];
const DECORATORS = ['Repeat', 'Retry', 'Timeout', 'Inverter', 'ForceSuccess', 'ForceFailure'];
// type: [[param, kind, default]...]; kind: num, angle, robots, name. No default: required.
const PARAMS = {
  Parallel: [['success', 'num', null], ['failure', 'num', 1]],
  Repeat: [['times', 'num', -1]],
  Retry: [['times', 'num', -1]],
  Timeout: [['seconds', 'num']],
  Wait: [['seconds', 'num']],
  GoTo: [['robot', 'robots'], ['x', 'num'], ['y', 'num'], ['yaw', 'angle', null]],
  Move: [['robot', 'robots'], ['forward', 'num', 0], ['left', 'num', 0], ['turn', 'angle', 0]],
  Approach: [['robot', 'robots'], ['target', 'name'], ['distance', 'num', 0.4]],
  Follow: [['robot', 'robots'], ['target', 'name'], ['distance', 'num', 0.4]],
  Stop: [['robot', 'robots']],
};
// The leaves (moves, Wait) and every node type, for the editor.
export const LEAVES = Object.keys(PARAMS).filter((t) => !COMPOSITES.includes(t) && !DECORATORS.includes(t));
export const TYPES = [...COMPOSITES, ...DECORATORS, ...LEAVES];
const ALIASES = { selector: 'Fallback', sleep: 'Wait', retryuntilsuccessful: 'Retry', goto: 'GoTo', invert: 'Inverter' };
// BehaviorTree.CPP's attribute names: [ours, scale].
const XML_PARAMS = {
  Wait: { msec: ['seconds', 1e-3] }, Timeout: { msec: ['seconds', 1e-3] },
  Repeat: { num_cycles: ['times', 1] }, Retry: { num_attempts: ['times', 1] },
  Parallel: { success_count: ['success', 1], failure_count: ['failure', 1] },
};

const typeOf = (word) => {
  const w = word.toLowerCase();
  return TYPES.find((t) => t.toLowerCase() === w) ?? ALIASES[w];
};

// An error of the tree's text; `line` (1-based) where it is, if known (the
// editor marks it).
const parseError = (msg, line = null) => Object.assign(new Error(msg), { line });

// A node: {type, params, children, where, line, text, status, state}.
function makeNode(type, raw, children, where, text, line = null) {
  const specs = PARAMS[type] ?? [];
  const params = {};
  const fail = (msg) => { throw parseError(`${where}: ${type}: ${msg}`, line); };
  for (const [k, v] of Object.entries(raw)) {
    const spec = specs.find(([p]) => p === k);
    if (!spec) fail(`no parameter ${k}${specs.length ? ` (${specs.map(([p]) => p).join(', ')})` : ''}`);
    params[k] = value(spec[1], v, (msg) => fail(`${k}: ${msg}`));
  }
  for (const [k, , d] of specs) {
    if (params[k] === undefined) {
      if (d === undefined) fail(`needs ${k}`);
      params[k] = d;
    }
  }
  const n = COMPOSITES.includes(type) ? [1, Infinity] : DECORATORS.includes(type) ? [1, 1] : [0, 0];
  if (children.length < n[0] || children.length > n[1]) {
    fail(n[1] === 0 ? 'has no children' : n[1] === 1 ? 'needs one child' : 'needs children');
  }
  return { type, params, children, where, line, text, status: IDLE, state: {} };
}

function value(kind, v, fail) {
  if (kind === 'robots') {
    const list = String(v).split(',').map((s) => s.trim()).filter(Boolean);
    if (!list.length) fail('empty');
    return list;
  }
  if (kind === 'name') return String(v);
  if (typeof v === 'number') return v;
  const s = String(v).trim();
  const deg = kind === 'angle' && /deg$/i.test(s);
  const x = Number(deg ? s.slice(0, -3) : s);
  if (s === '' || !Number.isFinite(x)) fail(`not a number: ${v}`);
  return deg ? x * Math.PI / 180 : x;
}

// The text form (`format` 'text') or BehaviorTree.CPP XML ('xml') -> the root node.
export function parse(text, format = 'text') {
  return format === 'xml' ? parseXml(text) : parseText(text);
}

function parseText(text) {
  const items = [];
  text.split('\n').forEach((raw, i) => {
    const line = raw.replace(/#.*/, '').replace(/\t/g, '  ');
    if (!line.trim()) return;
    const indent = line.length - line.trimStart().length;
    const [word, ...args] = line.trim().match(/"[^"]*"|'[^']*'|\S+/g);
    const where = `line ${i + 1}`;
    const type = typeOf(word);
    if (!type) throw parseError(`${where}: no node ${word} (${TYPES.join(', ')})`, i + 1);
    const raw2 = {}, specs = PARAMS[type] ?? [];
    let pos = 0;
    for (const a of args) {
      const m = a.match(/^(\w+)=(.*)$/);
      const unq = (s) => s.replace(/^(["'])(.*)\1$/, '$2');
      if (m) raw2[m[1]] = unq(m[2]);
      else {
        if (pos >= specs.length) throw parseError(`${where}: ${type}: too many parameters (${specs.map(([p]) => p).join(', ') || 'none'})`, i + 1);
        raw2[specs[pos++][0]] = unq(a);
      }
    }
    items.push({ indent, type, raw: raw2, where, line: i + 1, text: raw.trim().replace(/\s*#.*/, ''), children: [] });
  });
  if (!items.length) throw parseError('no nodes', 1);
  // Indentation -> children: a stack of the open nodes.
  const top = items[0];
  const stack = [top];
  for (const it of items.slice(1)) {
    while (stack.length && stack.at(-1).indent >= it.indent) stack.pop();
    if (!stack.length) throw parseError(`${it.where}: a second root (indent it under ${top.type})`, it.line);
    stack.at(-1).children.push(it);
    stack.push(it);
  }
  const build = (it) => makeNode(it.type, it.raw, it.children.map(build), it.where, it.text, it.line);
  return build(top);
}

function parseXml(text) {
  const doc = new DOMParser().parseFromString(text, 'application/xml');
  const err = doc.querySelector('parsererror');
  if (err) {
    const msg = err.querySelector('div')?.textContent ?? err.textContent;  // (Chrome: "error on line 3 at column 5: …")
    throw parseError(`XML: ${msg.trim().split('\n')[0]}`, Number(msg.match(/line (\d+)/)?.[1]) || null);
  }
  const root = doc.documentElement;
  const trees = root.tagName === 'BehaviorTree' ? [root] : [...root.children].filter((e) => e.tagName === 'BehaviorTree');
  const main = root.getAttribute('main_tree_to_execute');
  const tree = trees.find((t) => t.getAttribute('ID') === main) ?? trees[0];
  if (!tree || tree.children.length !== 1) throw new Error('XML: needs a <BehaviorTree> with one root node');
  const build = (e) => {
    const where = `<${e.tagName}>`;
    const type = typeOf(e.tagName);
    if (!type) throw new Error(`${where}: no node ${e.tagName} (${TYPES.join(', ')})`);
    const raw = {};
    for (const a of e.attributes) {
      if (a.name === 'name' || a.name === 'ID') continue;
      const map = XML_PARAMS[type]?.[a.name];
      raw[map ? map[0] : a.name] = map && map[1] !== 1 ? Number(a.value) * map[1] : a.value;
    }
    const text = [e.tagName, ...[...e.attributes].filter((a) => a.name !== 'name').map((a) => `${a.name}=${a.value}`)].join(' ');
    return makeNode(type, raw, [...e.children].map(build), where, e.getAttribute('name') || text);
  };
  return build(tree.children[0]);
}

// Every node of the tree, depth first.
export function nodes(n) {
  return [n, ...n.children.flatMap(nodes)];
}

// ---- running ----

function reset(n) {
  n.status = IDLE;
  n.state = {};
  n.children.forEach(reset);
}

function tick(n, ctx) {
  if (n.status !== RUNNING) {  // a fresh start: what ran below before is forgotten
    n.children.forEach(reset);
    n.state = {};
  }
  n.status = TICK[n.type](n, ctx);
  if (n.status !== RUNNING) HALT[n.type]?.(n, ctx);  // (a GoTo failing for one robot: the others' goals go)
  return n.status;
}

// Stops n if it runs (and what runs below it).
function halt(n, ctx) {
  if (n.status !== RUNNING) return;
  n.children.forEach((c) => halt(c, ctx));
  HALT[n.type]?.(n, ctx);
  n.status = IDLE;
}

const sequence = (on, other) => (n, ctx) => {
  const s = n.state;
  s.i ??= 0;
  while (s.i < n.children.length) {
    const r = tick(n.children[s.i], ctx);
    if (r === RUNNING) return RUNNING;
    if (r === other) return other;
    s.i++;
  }
  return on;
};
const reactive = (on) => (n, ctx) => {
  for (let i = 0; i < n.children.length; i++) {
    const r = tick(n.children[i], ctx);
    if (r === on) continue;
    n.children.slice(i + 1).forEach((c) => halt(c, ctx));
    return r;
  }
  return on;
};
const turned = (map) => (n, ctx) => map(tick(n.children[0], ctx));

const TICK = {
  Sequence: sequence(SUCCESS, FAILURE),
  Fallback: sequence(FAILURE, SUCCESS),
  ReactiveSequence: reactive(SUCCESS),
  ReactiveFallback: reactive(FAILURE),
  Parallel(n, ctx) {
    const s = n.state;
    s.done ??= new Map();  // child -> its result
    for (const c of n.children) if (!s.done.has(c)) { const r = tick(c, ctx); if (r !== RUNNING) s.done.set(c, r); }
    const count = (r) => [...s.done.values()].filter((x) => x === r).length;
    const need = n.params.success ?? n.children.length;
    const result = count(SUCCESS) >= need ? SUCCESS
      : count(FAILURE) >= n.params.failure || n.children.length - count(FAILURE) < need ? FAILURE : RUNNING;
    if (result !== RUNNING) n.children.forEach((c) => halt(c, ctx));
    return result;
  },
  Repeat(n, ctx) {
    const s = n.state, c = n.children[0];
    s.count ??= 0;
    const r = tick(c, ctx);
    if (r !== SUCCESS) return r;
    if (n.params.times >= 0 && ++s.count >= n.params.times) return SUCCESS;
    return RUNNING;  // the next round on the next tick
  },
  Retry(n, ctx) {
    const s = n.state, c = n.children[0];
    s.count ??= 0;
    const r = tick(c, ctx);
    if (r !== FAILURE) return r;
    if (n.params.times >= 0 && ++s.count >= n.params.times) return FAILURE;
    return RUNNING;
  },
  Timeout(n, ctx) {
    const s = n.state;
    if (s.t0 == null || ctx.time < s.t0) s.t0 = ctx.time;
    if (ctx.time - s.t0 >= n.params.seconds) { halt(n.children[0], ctx); return FAILURE; }
    return tick(n.children[0], ctx);
  },
  Inverter: turned((r) => (r === SUCCESS ? FAILURE : r === FAILURE ? SUCCESS : r)),
  ForceSuccess: turned((r) => (r === RUNNING ? r : SUCCESS)),
  ForceFailure: turned((r) => (r === RUNNING ? r : FAILURE)),
  Wait(n, ctx) {
    const s = n.state;
    if (s.t0 == null || ctx.time < s.t0) s.t0 = ctx.time;  // (time went back: a reset)
    const left = n.params.seconds - (ctx.time - s.t0);
    n.info = left > 0 ? `${left.toFixed(1)} s left` : '';
    return left <= 1e-9 ? SUCCESS : RUNNING;
  },
  GoTo: goalNode((n, r) => {
    const { x, y, yaw } = n.params;
    return { x, y, yaw: yaw ?? Math.atan2(y - r.pose.y, x - r.pose.x) };
  }),
  Move: goalNode((n, r) => {
    const { forward, left, turn } = n.params, { x, y, yaw } = r.pose, c = Math.cos(yaw), s = Math.sin(yaw);
    return { x: x + c * forward - s * left, y: y + s * forward + c * left, yaw: yaw + turn };
  }),
  Approach: followNode(false),
  Follow: followNode(true),
  Stop(n, ctx) {
    ctx.goals.clear(ctx.robots(n.params.robot).map((r) => r.name));
    return SUCCESS;
  },
};

const HALT = {
  GoTo: clearOwn, Move: clearOwn, Approach: clearOwn, Follow: clearOwn,
};

const same = (a, b) => !!a && !!b && ['x', 'y', 'yaw'].every((k) => Math.abs(a[k] - b[k]) < SAME);

// Sets goal(n, robot) for each robot when it starts; succeeds once all are reached.
function goalNode(goal) {
  return (n, ctx) => {
    const s = n.state;
    if (!s.set) {
      s.set = new Map();  // name -> the goal given
      for (const r of ctx.robots(n.params.robot)) {
        const g = goal(n, r);
        ctx.goals.set(r.name, g);
        s.set.set(r.name, g);
      }
    }
    return waitGoals(n, ctx);
  };
}

function waitGoals(n, ctx) {
  const s = n.state, info = [];
  let all = true;
  for (const [name, g] of s.set) {
    const r = ctx.robot(name);
    if (!r) { n.info = `${name} is gone`; return FAILURE; }
    if (r.goal == null && r.arrived && !s.reached?.has(name)) (s.reached ??= new Set()).add(name);
    if (s.reached?.has(name)) continue;
    if (!same(r.goal, g)) { n.info = `${name}'s goal was taken`; s.set.delete(name); return FAILURE; }
    all = false;
    info.push(`${name} ${Math.hypot(g.x - r.pose.x, g.y - r.pose.y).toFixed(2)} m`);
  }
  n.info = info.join(', ');
  return all ? SUCCESS : RUNNING;
}

// Approach (keep = false) or Follow (keep = true).
function followNode(keep) {
  return (n, ctx) => {
    const s = n.state;
    s.set ??= new Map();
    const robots = s.robots ??= ctx.robots(n.params.robot).map((r) => r.name);
    const t = ctx.thing(n.params.target);
    if (!t) { n.info = `no ${n.params.target}`; return FAILURE; }
    const { distance } = n.params, info = [];
    let all = true;
    for (const name of robots) {
      const r = ctx.robot(name);
      if (!r) { n.info = `${name} is gone`; return FAILURE; }
      if (r.name === t.name) { n.info = `${name} cannot follow itself`; return FAILURE; }
      const mine = s.set.get(name);
      if (mine && !r.goal && !r.arrived || mine && r.goal && !same(r.goal, mine)) {
        n.info = `${name}'s goal was taken`;
        s.set.delete(name);
        return FAILURE;
      }
      const dx = r.pose.x - t.pose.x, dy = r.pose.y - t.pose.y, d = Math.hypot(dx, dy) || 1;
      info.push(`${name} ${d.toFixed(2)} m`);
      if (!r.goal && d <= distance + NEAR) {  // there (Approach: done)
        s.set.delete(name);
        continue;
      }
      all = false;
      // On the line from the target to the robot, `distance` from it, facing it.
      const g = { x: t.pose.x + dx / d * distance, y: t.pose.y + dy / d * distance, yaw: Math.atan2(-dy, -dx) };
      if (r.goal && Math.hypot(r.goal.x - g.x, r.goal.y - g.y) < MOVED) continue;  // its goal still fits
      ctx.goals.set(name, g);
      s.set.set(name, g);
    }
    n.info = `${info.join(', ')} from ${t.name}`;
    return all && !keep ? SUCCESS : RUNNING;
  };
}

// The goals n set that the robots still have: cleared.
function clearOwn(n, ctx) {
  const names = [...(n.state.set ?? [])].filter(([name, g]) => same(ctx.robot(name)?.goal, g)).map(([name]) => name);
  if (names.length) ctx.goals.clear(names);
}

// What a tick sees, on `app` (its data.time) with `goals` (main.js's
// window.goals): goals.list() entries (as they are now: a goal just set
// shows) by name, robots by which.
export function context(app, goals) {
  const byName = (w) => {
    const list = goals.list(), part = w.toLowerCase();
    return list.find((r) => r.name === w) ?? list.find((r) => r.label === w)
      ?? list.find((r) => r.label.toLowerCase().includes(part));
  };
  return {
    time: app.data.time,
    goals,
    robot: (name) => goals.list().find((r) => r.name === name),
    thing: byName,
    robots: (which) => {
      const list = goals.list();
      if (which.length === 1 && which[0] === 'all') return list.filter((r) => r.drives);
      return [...new Set(which.flatMap((w) => {
        const rs = list.filter((r) => r.name === w || r.label === w);  // a label: all of them
        if (!rs.length) throw new Error(`no robot ${w}`);
        const off = rs.find((r) => !r.drives);
        if (off) throw new Error(`${off.name} cannot drive (no walking policy or base on)`);
        return rs;
      }))];
    },
  };
}

// One leaf (a move or Wait) on its own, for py_trees (pybt.js): params as
// in the text form (validated alike); tick() and halt() take a context().
export function leaf(type, params) {
  const t = typeOf(type);
  if (!t || COMPOSITES.includes(t) || DECORATORS.includes(t)) throw new Error(`no leaf ${type}`);
  const n = makeNode(t, params, [], t, t);
  return {
    tick: (ctx) => tick(n, ctx),
    halt: (ctx) => halt(n, ctx),
    get info() { return n.info ?? ''; },
  };
}

// A tree run on `app` (its data.time, paused) with `goals` (main.js's
// window.goals); `onTick(runner)` after each tick.
export class Runner {
  constructor(root, { app, goals, onTick }) {
    this.root = root;
    this.app = app;
    this.goals = goals;
    this.onTick = onTick;
    this.status = RUNNING;
    this.error = null;
    this.start = null;  // sim time it started
    this.timer = setInterval(() => this.step(), TICK_MS);
    this.step();
  }

  context() {
    return context(this.app, this.goals);
  }

  // The tree view's rows, depth first: [{depth, text, title, status, info}].
  rows() {
    const out = [];
    const add = (n, depth) => {
      out.push({ depth, text: n.text, title: n.where, status: n.status, info: n.status === IDLE ? '' : n.info ?? '' });
      n.children.forEach((c) => add(c, depth + 1));
    };
    add(this.root, 0);
    return out;
  }

  step() {
    if (this.status !== RUNNING || this.app.paused || !this.app.data) return;
    try {
      const ctx = this.context();
      this.start ??= ctx.time;
      if (ctx.time < this.start) this.start = ctx.time;
      this.time = ctx.time - this.start;
      const r = tick(this.root, ctx);
      if (r !== RUNNING) this.finish(r);
    } catch (err) {
      this.error = err.message ?? String(err);
      this.stop(FAILURE);
    }
    this.onTick?.(this);
  }

  finish(status) {
    clearInterval(this.timer);
    this.status = status;
  }

  // Stops the tree, clearing the goals its running nodes set.
  stop(status = IDLE) {
    if (this.status !== RUNNING) return;
    try { halt(this.root, this.context()); } catch { /* the world is gone */ }
    this.finish(status);
    this.onTick?.(this);
  }
}
