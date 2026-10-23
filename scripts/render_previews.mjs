// Renders <name>/<name>.jpg (the naming convention in scripts/vendor.mjs)
// for every asset (of the given categories or <category>/<source>es of
// mjcf_assets/) that has no screenshot yet, with the viewer itself in
// headless Chrome: each asset is spawned alone, framed by its collision
// geoms' bounds, and captured. A room (bodies named wall_*, as RoboCasa's
// kitchens) is shown cut away instead: framed by what is in it (its walls
// and floors left out), from above the side facing its main wall, with the
// walls on the camera's side hidden.
// Re-run node scripts/vendor.mjs to pick them up; scripts/thumbnails.py
// then turns them into panel thumbnails.
//
//   npm start &                                   # serves public/ on :4007
//   node scripts/render_previews.mjs [filter…]    # e.g. objects, robots/aicor,
//                                                 # objects/robocasa/apple; default: all
//   node scripts/vendor.mjs && python3 scripts/thumbnails.py
//
// Options (env): CHROME=/path/to/chrome (default: Playwright's headless
// shell in ~/.cache/ms-playwright), URL=http://127.0.0.1:4007/, FORCE=1 to
// re-render assets that already have a screenshot, GPU=1 to render on the
// GPU (Vulkan; several times faster than the default SwiftShader).
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';

const filters = process.argv.slice(2);
const url = process.env.URL ?? 'http://127.0.0.1:4007/';

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const cache = join(homedir(), '.cache/ms-playwright');
  for (const d of existsSync(cache) ? readdirSync(cache).sort().reverse() : []) {
    if (!d.startsWith('chromium')) continue;
    for (const sub of readdirSync(join(cache, d))) {
      for (const exe of ['chrome-headless-shell', 'chrome']) {
        if (existsSync(join(cache, d, sub, exe))) return join(cache, d, sub, exe);
      }
    }
  }
  throw new Error('no Chrome found: set CHROME=/path/to/chrome');
}

const index = 'public/json/assets.json';
if (!existsSync(index)) throw new Error(`no ${index}: run node scripts/vendor.mjs`);
const todo = [];
for (const m of JSON.parse(readFileSync(index, 'utf8'))) {
  if (m.screenshot && !process.env.FORCE) continue;
  // m.dir is 'mjcf_assets/<category dir>/<source>/<name>' (relative to public/,
  // where mjcf_assets is a symlink to the one next to package.json).
  // A filter also matches a model (<category>/<source>/<name>) or a group
  // (<category>/<source>/<group>, e.g. objects/robocasa/apple).
  const where = m.dir.split('/').slice(1).join('/');
  const group = `${m.dir.split('/').slice(1, 3).join('/')}/${m.group}`;
  if (filters.length && !filters.some((f) => [where, group].some((w) => w === f || w.startsWith(`${f}/`)))) continue;
  todo.push({ name: m.name, rel: `${m.dir}/${m.main}`, out: join(m.dir, `${m.name}.jpg`) });
}
console.log(`${todo.length} assets to render`);
if (!todo.length) process.exit(0);

const port = 9400 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), 'render-previews-'));
const chrome = spawn(findChrome(), [
  `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-sandbox', '--disable-dev-shm-usage',
  ...(process.env.GPU ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--ignore-gpu-blocklist']
                      : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader']),
  '--window-size=1280,800', 'about:blank',
], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50 && !targets; i++) {
  await sleep(200);
  targets = await fetch(`http://127.0.0.1:${port}/json`).then((r) => r.json()).catch(() => null);
}
const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
await new Promise((r) => { ws.onopen = r; });
let id = 0;
const pending = new Map();
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
};
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? 'evaluate failed');
  return r.result?.result?.value;
};

try {
  await send('Page.navigate', { url });
  for (let i = 0; i < 200 && !(await ev(`!!window.spawnAsset && !!window.app?.model`).catch(() => false)); i++) await sleep(300);
  await ev(`for (const id of ['ui', 'toggles', 'left', 'side', 'gizmo', 'stats']) document.getElementById(id).style.display = 'none'`);
  for (const [i, a] of todo.entries()) {
    await ev(`clearAssets()`);
    await ev(`spawnAsset(${JSON.stringify(a.rel)})`);
    if (!(await ev(`app.robots.length`))) { console.warn(`${a.name}: spawn failed`); continue; }
    await sleep(800);  // settle
    // Frame the collision geoms' bounds (body > 0, or the static ones of an
    // environment), from the default camera angle, then let textures load.
    await ev(`(() => {
      app.paused = true;
      app.setOriginMarker(false);
      if (app.teleop) app.teleop.active = null;  // no IK or goal markers
      const { model: m, data: d, mujoco: M } = app;
      const r = app.robots[0];
      const name = (type, i) => (M.mj_id2name(m, M.mjtObj[type].value, i) ?? '').replace(r.prefix, '');
      const shell = (b) => /^(wall|floor)_(?!accessory)/.test(name('mjOBJ_BODY', b));  // a room's walls, floors
      const ours = (b) => b >= r.bodyOffset && b < r.bodyOffset + r.visual.inertia.nbody - 1;
      const mainWall = (b) => /^wall_(main|room)/.test(name('mjOBJ_BODY', b));  // RoboCasa: wall_main_* or wall_room_*
      const room = Array.from({ length: m.nbody }, (_, b) => b).some((b) => ours(b) && mainWall(b));
      // World bounds of geoms (their aabb turned into the world) that pass use.
      const bounds = (use) => {
        const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (let g = 1; g < m.ngeom; g++) {
          if (!m.geom_rbound[g] || !use(g)) continue;
          const a = m.geom_aabb.subarray(6 * g, 6 * g + 6), R = d.geom_xmat.subarray(9 * g, 9 * g + 9);
          for (let k = 0; k < 3; k++) {
            let c = d.geom_xpos[3 * g + k], h = 0;
            for (let j = 0; j < 3; j++) { c += R[3 * k + j] * a[j]; h += Math.abs(R[3 * k + j]) * a[3 + j]; }
            lo[k] = Math.min(lo[k], c - h);
            hi[k] = Math.max(hi[k], c + h);
          }
        }
        return [lo, hi];
      };
      // A room: what is inside its walls and floors (RoboCasa parks unused
      // fixture parts 10 m up).
      const [rlo, rhi] = room ? bounds((g) => ours(m.geom_bodyid[g]) && shell(m.geom_bodyid[g])) : [];
      const inside = (g) => [0, 1, 2].every((k) => d.geom_xpos[3 * g + k] > rlo[k] && d.geom_xpos[3 * g + k] < rhi[k]);
      const [lo, hi] = bounds((g) => !room || (ours(m.geom_bodyid[g]) && !shell(m.geom_bodyid[g]) && inside(g)));
      const c = lo.map((v, k) => (v + hi[k]) / 2), radius = 0.5 * Math.hypot(...hi.map((v, k) => v - lo[k]));
      app.orbit.center = c;
      app.orbit.azim = -0.6; app.orbit.elev = 0.45;
      app.orbit.dist = Math.max(0.15, 0.85 * radius / Math.tan(22.5 * Math.PI / 180));
      if (!room) return;
      // From the side away from the main wall, a quarter turn off square.
      const main = [0, 0];
      for (let b = 1; b < m.nbody; b++) {
        if (ours(b) && mainWall(b)) { main[0] = d.xpos[3 * b]; main[1] = d.xpos[3 * b + 1]; break; }
      }
      app.orbit.azim = Math.atan2(c[1] - main[1], c[0] - main[0]) - 0.35;
      app.orbit.elev = 0.75;
      app.orbit.dist = 0.95 * radius / Math.tan(22.5 * Math.PI / 180);
      // Hide the walls between the camera and the middle.
      const eye = [Math.cos(app.orbit.azim), Math.sin(app.orbit.azim)];
      for (const [i, part] of r.visual.parts.entries()) {
        const b = r.bodyOffset + part.body - 1;
        if (part.body <= 0 || !/^wall_(?!accessory)/.test(name('mjOBJ_BODY', b))) continue;
        if ((d.xpos[3 * b] - c[0]) * eye[0] + (d.xpos[3 * b + 1] - c[1]) * eye[1] > 0.3) app.scene.remove(r.entities[i]);
      }
    })()`);
    await sleep(1500);
    // 720 px of the page, saved 500 px wide.
    const shot = await send('Page.captureScreenshot', { format: 'jpeg', quality: 85, clip: { x: 280, y: 40, width: 720, height: 720, scale: 500 / 720 } });
    writeFileSync(a.out, Buffer.from(shot.result.data, 'base64'));
    await ev(`app.paused = false`);
    console.log(`[${i + 1}/${todo.length}] ${a.out}`);
  }
} finally {
  chrome.kill();
  await sleep(300);
  rmSync(profile, { recursive: true, force: true });
}
process.exit(0);
