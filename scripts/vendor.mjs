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

// Model libraries checked out next to package.json: each is exposed as
// public/<name> (symlink; http.server follows it) and its models are listed
// in public/<name>.json for the asset panel, one entry per model directory
// with a scene.xml.
//
// `main` is the model file scene.xml includes: spawning that instead of the
// scene keeps the scene's own floor and lights out of the shared world. A
// scene.xml without includes (a whole environment, say) is spawned as is.
//
// `category` (the asset panel tab: environment, robot or object) is the
// library's default, or for 'auto' "robot" if scene.xml includes a model file
// and "environment" otherwise. `screenshot` is the image in the model
// directory that scripts/thumbnails.py turns into the panel thumbnail: the one
// named like `main` or the directory, else one with "preview" in its name,
// else the shortest name. An asset.json in the model directory, e.g.
// {"category": "object", "main": "mug.xml", "thumbnail": "mug.png"},
// overrides these.
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

function indexLibrary(dir, name, category, defaultGroup = null) {
  if (!existsSync(dir)) return;
  const link = `public/${name}`;
  if (!existsSync(link) && !lstatSync(link, { throwIfNoEntry: false })) symlinkSync(`../${dir}`, link);

  const groups = readmeGroups(dir);
  const models = [];
  for (const d of readdirSync(dir, { withFileTypes: true })) {
    if (!d.isDirectory() || d.name.startsWith('.')) continue;
    const scene = `${dir}/${d.name}/scene.xml`, meta = `${dir}/${d.name}/asset.json`;
    const over = existsSync(meta) ? JSON.parse(readFileSync(meta, 'utf8')) : {};
    if (!existsSync(scene) && !over.main) continue;
    const included = existsSync(scene) && readFileSync(scene, 'utf8').match(/<include\s+file="([^"]+\.xml)"/)?.[1];
    models.push({
      name: d.name,
      main: over.main ?? included ?? 'scene.xml',
      category: over.category ?? (category === 'auto' ? (included ? 'robot' : 'environment') : category),
    });
    const shot = over.thumbnail ?? pickScreenshot(`${dir}/${d.name}`, models.at(-1).main, d.name);
    if (shot) models.at(-1).screenshot = shot;
    const group = over.group ?? groups.get(d.name)?.[0] ?? defaultGroup;
    if (group) models.at(-1).group = group;
    models.at(-1).rank = over.group || defaultGroup ? -1 : groups.get(d.name)?.[1] ?? Infinity;
  }
  // By group (in README order; asset.json groups first, ungrouped last), then
  // name: the asset panel lists groups in the order they first appear.
  models.sort((a, b) => a.rank - b.rank || (a.group ?? '').localeCompare(b.group ?? '') || a.name.localeCompare(b.name));
  for (const m of models) delete m.rank;
  writeFileSync(`public/${name}.json`, JSON.stringify(models, null, 1));
  console.log('indexed', models.length, `${dir} models into public/${name}.json`);
}
indexLibrary('my_robot', 'my_robot', 'robot', 'My robots');  // your own robots: listed first
indexLibrary('mujoco_menagerie', 'menagerie', 'robot');
indexLibrary('my_environment', 'my_environment', 'auto');  // your own models
indexLibrary('my_objects', 'my_objects', 'object');        // your own objects, if any

// If the GR00T policies (huggingface.co/nepyope/GR00T-WholeBodyControl_g1) are
// checked out next to package.json, expose them as public/groot: spawned
// unitree_g1 robots then walk under that policy (public/groot.js).
const GROOT = 'GR00T-WholeBodyControl_g1';
if (existsSync(GROOT)) {
  const link = 'public/groot';
  if (!existsSync(link) && !lstatSync(link, { throwIfNoEntry: false })) symlinkSync(`../${GROOT}`, link);
  console.log('linked', GROOT, 'as', link);
}
