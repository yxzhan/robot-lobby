// Behaviours for many robots at once, from the devtools console, built on
// main.js's window.goals (so they work in local and network mode alike, the
// goals showing as markers): each runs on a timer, setting the goals of the
// robots it drives, until stopped.
//
//   goals.randomWalk({ model: 'microduck', exclude: ['r1'] })
//   goals.chase({ target: 'r1', distance: 0.4 })   // the others walk after r1
//   goals.running()                                // [{id, kind, robots, opts}]
//   goals.stop()                                   // all of them (or goals.stop(id))
//
// Robots taken (every option): `robots` ('all': every robot whose walking
// policy or base is on, a name 'r3', a label, or a list) whose label
// contains `model` (case-insensitive, '' for any), minus the names in
// `exclude`; robots spawned later join on the next tick. A robot that two
// behaviours take is driven by the one started last. Stopping one clears the
// goals of the robots it drove.

const COMMON = {
  robots: 'all',  // as in goals.move
  model: '',      // part of the label, as in the Stage panel
  exclude: [],    // names left alone, e.g. ['r1', 'r4']
};

const CHASE = {
  target: 'r1',   // name (or label) of the robot chased; never a chaser
  distance: 0.4,  // m, how far from the target a chaser stops
  interval: 0.5,  // s between goal updates
  moved: 0.1,     // m, a goal is set again only once it would move this far
};

const RANDOM_WALK = {
  interval: 3,    // s between rounds
  min: 1,         // robots given a goal per round, at least
  max: 3,         // and at most (capped by how many there are)
  // Each a number x (from -x to x; distance: from 0 to x) or a range [min, max]:
  distance: [0.3, 1.0],  // m, how far ahead a goal is (< 0: behind)
  side: 0.3,      // m, to the left (> 0) or right
  turn: Math.PI / 2,  // rad, how much the goal heading turns (> 0: left)
  radius: 3,      // m, goals stay within this of `center`
  center: null,   // [x, y] of the play area; null: the robots' middle at start
  busy: false,    // also pick robots that are still on their way
  stuck: 8,       // s: a goal not reached by then counts as given up
};

const rand = (a, b) => a + Math.random() * (b - a);
// A number or [min, max] as a range ([0, x] or [-x, x] for a number).
const range = (v, from0 = false) => Array.isArray(v) ? v.map(Number) : [from0 ? 0 : -v, +v];
const shuffle = (xs) => {
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
  }
  return xs;
};

// Adds chase, randomWalk, running and stop to `goals` (main.js).
export function addSwarm(goals) {
  const running = [];  // started order: {id, kind, opts, timer, take(), owned}
  let nextId = 1;

  // The robots of the options (goals.list() entries), before ownership.
  const select = (opts) => {
    const list = goals.list(), model = opts.model.toLowerCase();
    let names;
    if (opts.robots === 'all') names = new Set(list.filter((r) => r.drives).map((r) => r.name));
    else {
      const ws = [opts.robots].flat().map(String);
      names = new Set(list.filter((r) => ws.includes(r.name) || ws.includes(r.label)).map((r) => r.name));
    }
    return list.filter((r) => r.drives && names.has(r.name) && r.label.toLowerCase().includes(model)
      && !opts.exclude.includes(r.name) && !opts.skip?.(r));
  };
  // Those of `b` that no later behaviour takes.
  const own = (b) => {
    const later = running.slice(running.indexOf(b) + 1).map((x) => new Set(select(x.opts).map((r) => r.name)));
    const mine = select(b.opts).filter((r) => !later.some((s) => s.has(r.name)));
    for (const r of mine) b.owned.add(r.name);
    return mine;
  };

  const start = (kind, defaults, options, setup, tick) => {
    const opts = { ...COMMON, ...defaults, ...options };
    opts.exclude = [opts.exclude].flat().map(String);
    const b = { id: nextId++, kind, opts, timer: null, owned: new Set() };
    setup?.(opts);
    running.push(b);
    const step = () => {
      try { tick(own(b), opts); } catch (err) { console.warn(`[${kind} ${b.id}]`, err); }
    };
    step();
    b.timer = setInterval(step, opts.interval * 1000);
    console.log(`[${kind} ${b.id}] started on ${own(b).map((r) => r.name).join(', ') || 'no robot yet'}`, opts);
    return b.id;
  };

  goals.chase = (options = {}) => start('chase', CHASE, options, (opts) => {
    opts.target = String(opts.target);
    Object.defineProperty(opts, 'skip', { value: (r) => r.name === opts.target || r.label === opts.target });  // not logged
  }, (robots, opts) => {
    const t = goals.list().find((r) => r.name === opts.target || r.label === opts.target);
    if (!t) return;
    for (const r of robots) {
      const dx = r.pose.x - t.pose.x, dy = r.pose.y - t.pose.y, d = Math.hypot(dx, dy) || 1;
      // On the line from the target to the chaser, `distance` from the target, facing it.
      const x = t.pose.x + dx / d * opts.distance, y = t.pose.y + dy / d * opts.distance;
      if (d <= opts.distance + opts.moved && !r.goal) continue;  // close enough already
      if (r.goal && Math.hypot(r.goal.x - x, r.goal.y - y) < opts.moved) continue;  // its goal still fits
      goals.set(r.name, { x, y, yaw: Math.atan2(-dy, -dx) });
    }
  });

  goals.randomWalk = (options = {}) => {
    const since = new Map();  // name -> when it was last given a goal here (ms)
    return start('randomWalk', RANDOM_WALK, options, (opts) => {
      for (const k of ['distance', 'side', 'turn']) {
        if (range(opts[k]).some((v) => !Number.isFinite(v))) throw new Error(`randomWalk: ${k} is a number or [min, max]`);
      }
      if (!opts.center) {
        const ps = select(opts).map((r) => r.pose);
        opts.center = ps.length ? [0, 1].map((k) => ps.reduce((a, p) => a + [p.x, p.y][k], 0) / ps.length) : [0, 0];
      }
    }, (robots, opts) => {
      const free = robots.filter((r) => opts.busy || !r.goal || performance.now() - (since.get(r.name) ?? 0) > opts.stuck * 1000);
      const n = Math.min(free.length, opts.min + Math.floor(Math.random() * (opts.max - opts.min + 1)));
      const [cx, cy] = opts.center;
      for (const r of shuffle(free).slice(0, n)) {
        since.set(r.name, performance.now());
        const forward = rand(...range(opts.distance, true)), left = rand(...range(opts.side)), turn = rand(...range(opts.turn));
        const { x, y, yaw } = r.pose, c = Math.cos(yaw), s = Math.sin(yaw);
        if (Math.hypot(x + c * forward - s * left - cx, y + s * forward + c * left - cy) <= opts.radius) {
          goals.move(r.name, { forward, left, turn });
        } else {
          // Out of the area: a goal somewhere inside it, facing the way there.
          const a = rand(0, 2 * Math.PI), d = opts.radius * Math.sqrt(Math.random()) * 0.8;
          const tx = cx + d * Math.cos(a), ty = cy + d * Math.sin(a);
          goals.set(r.name, { x: tx, y: ty, yaw: Math.atan2(ty - y, tx - x) });
        }
      }
    });
  };

  goals.running = () => running.map((b) => ({ id: b.id, kind: b.kind, robots: own(b).map((r) => r.name), opts: b.opts }));

  // Stops behaviour `id` (all without one) and clears its robots' goals.
  goals.stop = (id) => {
    const stopped = running.filter((b) => id == null || b.id === id);
    if (id != null && !stopped.length) throw new Error(`no behaviour ${id}`);
    for (const b of stopped) {
      clearInterval(b.timer);
      running.splice(running.indexOf(b), 1);
    }
    // Not those another one still drives.
    const kept = new Set(running.flatMap((b) => own(b).map((r) => r.name)));
    const present = new Set(goals.list().map((r) => r.name));
    const names = [...new Set(stopped.flatMap((b) => [...b.owned]))].filter((n) => present.has(n) && !kept.has(n));
    if (names.length) goals.clear(names);
    for (const b of stopped) console.log(`[${b.kind} ${b.id}] stopped`);
    return stopped.map((b) => b.id);
  };
}
