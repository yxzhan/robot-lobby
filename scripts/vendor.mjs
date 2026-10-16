// Copies the MuJoCo and Filament runtime files out of node_modules so that
// public/ is a self-contained static site (no bundler needed).
import { cpSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs';

const copies = {
  'node_modules/@mujoco/mujoco/mujoco.js': 'public/vendor/mujoco/mujoco.js',
  'node_modules/@mujoco/mujoco/mujoco.wasm': 'public/vendor/mujoco/mujoco.wasm',
  'node_modules/filament/filament.js': 'public/vendor/filament/filament.js',
  'node_modules/filament/filament.wasm': 'public/vendor/filament/filament.wasm',
  // onnxruntime-web, WASM backend only (runs the GR00T policy, public/groot.js).
  'node_modules/onnxruntime-web/dist/ort.wasm.min.mjs': 'public/vendor/onnxruntime/ort.wasm.min.mjs',
  'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs': 'public/vendor/onnxruntime/ort-wasm-simd-threaded.mjs',
  'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm': 'public/vendor/onnxruntime/ort-wasm-simd-threaded.wasm',
  // mrdoob's FPS / frame time / memory meter (ES module).
  'node_modules/stats.js/src/Stats.js': 'public/vendor/stats/Stats.js',
};
for (const [from, to] of Object.entries(copies)) {
  mkdirSync(to.slice(0, to.lastIndexOf('/')), { recursive: true });
  cpSync(from, to);
}
console.log('vendored', Object.keys(copies).length, 'files into public/vendor');

// The model library: mjcf_assets/<category>/<source>/<model>/, exposed as
// public/mjcf_assets (symlink; the server follows it) and indexed into
// public/assets.json for the asset panel. <category> is the panel tab
// (environments, robots, objects); <source> where the models come from
// (digital_twin, mujoco_menagerie, robocasa, …). A model is a directory with
// a scene.xml (or an asset.json naming its entry file).
//
// Each entry: {name, dir, main, category, source, group, screenshot}, where
// `dir` is the model directory relative to public/ and `main` the file to
// spawn in it: the model file scene.xml includes (so the scene's own floor
// and lights stay out of the shared world), or scene.xml itself if it
// includes none (a whole environment). `group` is the header the panel lists
// it under: the section of menagerie's README for menagerie models, the
// source otherwise. `screenshot` (relative to `dir`) is the image that
// scripts/thumbnails.py turns into the panel thumbnail: the one named like
// `main` or the directory, else one with "preview" in its name, else the
// shortest name. An asset.json in the model directory, e.g.
// {"main": "mug.xml", "thumbnail": "mug.png", "group": "Kitchen"},
// overrides these.
//
// Order (the panel keeps it): digital_twin first, then the other sources by
// name; menagerie models by README section; then by name.
const LIBRARY = 'mjcf_assets';
const CATEGORIES = { environments: 'environment', robots: 'robot', objects: 'object' };
const FIRST_SOURCES = ['digital_twin'];

function pickScreenshot(path, main, name) {
  const images = readdirSync(path).filter((f) => /\.(png|jpe?g|webp)$/i.test(f));
  const stem = (f) => f.replace(/\.[^.]*$/, '').toLowerCase();
  return images.find((f) => stem(f) === stem(main)) ?? images.find((f) => stem(f) === name.toLowerCase())
    ?? images.find((f) => stem(f).includes('preview'))
    ?? images.sort((a, b) => a.length - b.length || a.localeCompare(b))[0];
}

// Menagerie's README lists its models by kind ("**Humanoids.**", "**Arms.**",
// …) in the "## Menagerie Models" section, each row linking to
// <model dir>/<scene>.xml: model dir -> [group, rank of the group]. A model
// listed twice (e.g. an arm whose gripper also has its own row) keeps its
// first group.
function readmeGroups(dir) {
  const groups = new Map();
  const readme = `${dir}/README.md`;
  if (!existsSync(readme)) return groups;
  const text = readFileSync(readme, 'utf8');
  const start = text.indexOf('## Menagerie Models');
  if (start < 0) return groups;
  let group = null, rank = -1;
  for (const line of text.slice(start).split('\n').slice(1)) {
    if (line.startsWith('## ')) break;
    const heading = line.match(/^\*\*(.+?)\.?\*\*\s*$/);
    if (heading) { group = heading[1]; rank++; continue; }
    const model = group && line.match(/mujoco_menagerie\/main\/([^/]+)\//)?.[1];
    if (model && !groups.has(model)) groups.set(model, [group, rank]);
  }
  return groups;
}

const subdirs = (dir) => readdirSync(dir, { withFileTypes: true })
  .filter((d) => d.isDirectory() && !d.name.startsWith('.')).map((d) => d.name);

function indexLibrary() {
  if (!existsSync(LIBRARY)) {
    console.warn(`no ${LIBRARY}/: the asset panel will be empty`);
    writeFileSync('public/assets.json', '[]');
    return;
  }
  const link = `public/${LIBRARY}`;
  if (!existsSync(link) && !lstatSync(link, { throwIfNoEntry: false })) symlinkSync(`../${LIBRARY}`, link);

  const models = [];
  for (const catDir of subdirs(LIBRARY)) {
    const category = CATEGORIES[catDir];
    if (!category) { console.warn(`skipping ${LIBRARY}/${catDir}: not one of ${Object.keys(CATEGORIES)}`); continue; }
    for (const source of subdirs(`${LIBRARY}/${catDir}`)) {
      const sourceDir = `${LIBRARY}/${catDir}/${source}`;
      const groups = readmeGroups(sourceDir);
      for (const name of subdirs(sourceDir)) {
        const dir = `${sourceDir}/${name}`;
        const scene = `${dir}/scene.xml`, meta = `${dir}/asset.json`;
        const over = existsSync(meta) ? JSON.parse(readFileSync(meta, 'utf8')) : {};
        if (!existsSync(scene) && !over.main) continue;
        const included = existsSync(scene) && readFileSync(scene, 'utf8').match(/<include\s+file="([^"]+\.xml)"/)?.[1];
        const main = over.main ?? included ?? 'scene.xml';
        const m = { name, dir, main, category, source, group: over.group ?? groups.get(name)?.[0] ?? source };
        const shot = over.thumbnail ?? pickScreenshot(dir, main, name);
        if (shot) m.screenshot = shot;
        m.rank = [FIRST_SOURCES.includes(source) ? FIRST_SOURCES.indexOf(source) : FIRST_SOURCES.length, source,
                  over.group ? -1 : groups.get(name)?.[1] ?? Infinity, m.group, name];
        models.push(m);
      }
    }
  }
  const cmp = (a, b) => (typeof a === 'string' ? a.localeCompare(b) : a - b);
  models.sort((a, b) => a.rank.reduce((c, v, k) => c || cmp(v, b.rank[k]), 0));
  for (const m of models) delete m.rank;
  writeFileSync('public/assets.json', JSON.stringify(models, null, 1));
  const count = {};
  for (const m of models) count[`${m.category}/${m.source}`] = (count[`${m.category}/${m.source}`] ?? 0) + 1;
  console.log('indexed', models.length, `models of ${LIBRARY}/ into public/assets.json:`, count);
}
indexLibrary();

// If the GR00T policies (huggingface.co/nepyope/GR00T-WholeBodyControl_g1) are
// checked out next to package.json, expose them as public/groot: spawned
// unitree_g1 robots then walk under that policy (public/groot.js).
const GROOT = 'GR00T-WholeBodyControl_g1';
if (existsSync(GROOT)) {
  const link = 'public/groot';
  if (!existsSync(link) && !lstatSync(link, { throwIfNoEntry: false })) symlinkSync(`../${GROOT}`, link);
  console.log('linked', GROOT, 'as', link);
}
