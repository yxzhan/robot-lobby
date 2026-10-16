// Renders <name>/<name>.png (the naming convention in scripts/vendor.mjs)
// for every asset (of the given categories or <category>/<source>es of
// mjcf_assets/) that has no screenshot yet, with the viewer itself in
// headless Chrome: each asset is spawned alone, framed by its collision
// geoms' bounds, and captured.
// Re-run node scripts/vendor.mjs to pick them up; scripts/thumbnails.py
// then turns them into panel thumbnails.
//
//   npm start &                                   # serves public/ on :4000
//   node scripts/render_previews.mjs [filter…]    # e.g. objects, robots/digital_twin; default: all
//   node scripts/vendor.mjs && python3 scripts/thumbnails.py
//
// Options (env): CHROME=/path/to/chrome (default: Playwright's headless
// shell in ~/.cache/ms-playwright), URL=http://127.0.0.1:4000/, FORCE=1 to
// re-render assets that already have a screenshot.
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';

const filters = process.argv.slice(2);
const url = process.env.URL ?? 'http://127.0.0.1:4000/';

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

const index = 'public/assets.json';
if (!existsSync(index)) throw new Error(`no ${index}: run node scripts/vendor.mjs`);
const todo = [];
for (const m of JSON.parse(readFileSync(index, 'utf8'))) {
  if (m.screenshot && !process.env.FORCE) continue;
  // m.dir is 'mjcf_assets/<category dir>/<source>/<name>' (relative to public/,
  // where mjcf_assets is a symlink to the one next to package.json).
  const where = m.dir.split('/').slice(1, 3).join('/');
  if (filters.length && !filters.some((f) => where === f || where.startsWith(`${f}/`))) continue;
  todo.push({ name: m.name, rel: `${m.dir}/${m.main}`, out: join(m.dir, `${m.name}.png`) });
}
console.log(`${todo.length} assets to render`);
if (!todo.length) process.exit(0);

const port = 9400 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), 'render-previews-'));
const chrome = spawn(findChrome(), [
  `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-sandbox', '--disable-dev-shm-usage',
  '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--window-size=1280,800', 'about:blank',
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
  await ev(`for (const id of ['ui', 'views', 'assets', 'side', 'gizmo', 'stats']) document.getElementById(id).style.display = 'none'`);
  for (const [i, a] of todo.entries()) {
    await ev(`clearAssets()`);
    await ev(`spawnAsset(${JSON.stringify(a.rel)})`);
    if (!(await ev(`app.robots.length`))) { console.warn(`${a.name}: spawn failed`); continue; }
    await sleep(800);  // settle
    // Frame the collision geoms' bounds (body > 0, or the static ones of an
    // environment), from the default camera angle, then let textures load.
    await ev(`(() => {
      app.paused = true;
      const { model: m, data: d } = app;
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let g = 1; g < m.ngeom; g++) {
        const r = m.geom_rbound[g];
        if (!r) continue;
        for (let k = 0; k < 3; k++) {
          lo[k] = Math.min(lo[k], d.geom_xpos[3 * g + k] - r);
          hi[k] = Math.max(hi[k], d.geom_xpos[3 * g + k] + r);
        }
      }
      const c = lo.map((v, k) => (v + hi[k]) / 2), radius = 0.5 * Math.hypot(...hi.map((v, k) => v - lo[k]));
      app.orbit.center = c;
      app.orbit.azim = -0.6; app.orbit.elev = 0.45;
      app.orbit.dist = Math.max(0.15, 0.85 * radius / Math.tan(22.5 * Math.PI / 180));
    })()`);
    await sleep(1500);
    const shot = await send('Page.captureScreenshot', { format: 'png', clip: { x: 280, y: 40, width: 720, height: 720, scale: 1 } });
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
