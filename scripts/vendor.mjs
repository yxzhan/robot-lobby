// Copies the MuJoCo and Filament runtime files out of node_modules so that
// public/ is a self-contained static site (no bundler needed).
import { cpSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, symlinkSync, unlinkSync, writeFileSync } from 'node:fs';

const copies = {
  'node_modules/@mujoco/mujoco/mujoco.js': 'public/vendor/mujoco/mujoco.js',
  'node_modules/@mujoco/mujoco/mujoco.wasm': 'public/vendor/mujoco/mujoco.wasm',
  'node_modules/filament/filament.js': 'public/vendor/filament/filament.js',
  'node_modules/filament/filament.wasm': 'public/vendor/filament/filament.wasm',
  // onnxruntime-web, WASM backend only (runs the ONNX policies, public/policy.js).
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
// (digital_twin, mujoco_menagerie, robocasa, …).
//
// Naming convention: model directory <name>/ holds <name>.xml (the file that
// gets spawned) and <name>.png (its screenshot, optional; scripts/
// thumbnails.py makes the panel thumbnail from it). Directories without
// <name>.xml are skipped with a warning. The one exception is upstream
// mujoco_menagerie, which keeps its own file names and is indexed from its
// README's model table instead (menagerieModels).
//
// Each entry: {name, dir, main, category, source, group, screenshot}, `dir`
// relative to public/, `screenshot` relative to `dir`. `group` is the header
// the panel lists it under: the section of menagerie's README (or "Others")
// for menagerie models, the source otherwise; "group" in an optional
// asset.json in the model directory overrides it.
//
// Order (the panel keeps it): digital_twin first, then the other sources by
// name; menagerie models by README section, "Others" last; then by name.
const LIBRARY = 'mjcf_assets';
const CATEGORIES = { environments: 'environment', robots: 'robot', objects: 'object' };
const FIRST_SOURCES = ['digital_twin'];
const MENAGERIE = 'mujoco_menagerie';

const includedFile = (path) => readFileSync(path, 'utf8').match(/<include\s+file="([^"]+\.xml)"/)?.[1] ?? null;

// Menagerie's README lists its models by kind ("**Humanoids.**", "**Arms.**",
// …) in the "## Menagerie Models" table; each row links to <dir>/<file>.xml
// (a scene, or for a few models the model itself: franka_emika_panda/
// hand.xml, realsense_d435i/d435i.xml) and shows assets/<…>.png. Returns the
// rows in order: [{dir, file, img, group, rank (of the group)}].
function readmeRows(sourceDir) {
  const readme = `${sourceDir}/README.md`;
  if (!existsSync(readme)) return [];
  const text = readFileSync(readme, 'utf8');
  const start = text.indexOf('## Menagerie Models');
  if (start < 0) return [];
  const rows = [];
  let group = null, rank = -1;
  for (const line of text.slice(start).split('\n').slice(1)) {
    if (line.startsWith('## ')) break;
    const heading = line.match(/^\*\*(.+?)\.?\*\*\s*$/);
    if (heading) { group = heading[1]; rank++; continue; }
    const link = group && line.match(/mujoco_menagerie\/main\/([^/]+)\/([^'"]+\.xml)/);
    if (link) rows.push({ dir: link[1], file: link[2], img: line.match(/<img src='([^']+)'/)?.[1], group, rank });
  }
  return rows;
}

// Menagerie keeps upstream file names, so its models come from the README
// table, one per row: the entry is the model file the linked scene includes
// (or the linked file itself if it includes none), the screenshot the row's
// image. A directory listed twice (an arm and its gripper) gives a second
// model named <dir>_<file>. Then every other scene_<suffix>.xml of a
// directory (variants: scene_right.xml, scene_with_hands.xml, scene_motor.xml,
// …) adds the model it includes as <dir>_<suffix>, in the same group, with
// the matching .png (<model>.png, <base model>_<suffix>.png or
// <dir>_<suffix>.png), else the directory's only .png, else the base
// model's screenshot. Directories the table does not list but that have a
// scene.xml including a model go under "Others". MJX models (scene or model
// file named *mjx*) are left out: they collide only through contact pairs
// with their scene's own floor, so they fall through the viewer's.
const MJX = /mjx/i;
function menagerieModels(sourceDir) {
  const models = [], byDir = new Map();  // dir -> its models
  const add = (m) => {
    if (MJX.test(m.scene) || MJX.test(m.main)) return;
    models.push(m);
    if (!byDir.has(m.dirName)) byDir.set(m.dirName, []);
    byDir.get(m.dirName).push(m);
  };
  for (const row of readmeRows(sourceDir)) {
    const dir = `${sourceDir}/${row.dir}`, linked = `${dir}/${row.file}`;
    if (!existsSync(linked)) { console.warn(`skipping ${linked}: listed in the README but missing`); continue; }
    const main = includedFile(linked) ?? row.file;
    const name = byDir.has(row.dir) ? `${row.dir}_${row.file.replace(/\.xml$/, '').replace(/^scene_?/, '') || 'scene'}` : row.dir;
    const screenshot = row.img && existsSync(`${sourceDir}/${row.img}`) ? `../${row.img}` : undefined;
    add({ name, dirName: row.dir, dir, main, scene: row.file, group: row.group, rank: row.rank, screenshot });
  }
  const stem = (f) => f.replace(/\.[^.]*$/, '');
  for (const dirName of subdirs(sourceDir)) {
    const dir = `${sourceDir}/${dirName}`;
    const files = readdirSync(dir);
    const pngs = files.filter((f) => f.toLowerCase().endsWith('.png'));
    // Shortest first (scene.xml, then scene_motor.xml before scene_motor_2.xml):
    // when two scenes include the same model, the plainer name keeps it.
    const scenes = files.filter((f) => /^scene(_.+)?\.xml$/.test(f)).sort((a, b) => a.length - b.length || a.localeCompare(b));
    for (const scene of scenes) {
      const known = byDir.get(dirName) ?? [];
      const main = includedFile(`${dir}/${scene}`);
      if (!main || known.some((m) => m.scene === scene || m.main === main)) continue;
      const base = known[0];
      if (!base) {  // not in the README table
        const shot = [`${dirName}.png`, `${stem(main)}.png`].find((f) => pngs.includes(f)) ?? (pngs.length === 1 ? pngs[0] : undefined);
        add({ name: dirName, dirName, dir, main, scene, group: 'Others', rank: Infinity, screenshot: shot });
        continue;
      }
      const suffix = stem(scene).replace(/^scene_/, '');
      const name = `${dirName}_${suffix}`;
      const shot = [`${stem(main)}.png`, `${stem(base.main)}_${suffix}.png`, `${name}.png`].find((f) => pngs.includes(f))
        ?? (pngs.length === 1 ? pngs[0] : base.screenshot);
      add({ name, dirName, dir, main, scene, group: base.group, rank: base.rank, screenshot: shot });
    }
  }
  return models;
}

const subdirs = (dir) => readdirSync(dir, { withFileTypes: true })
  .filter((d) => d.isDirectory() && !d.name.startsWith('.')).map((d) => d.name);

// Models following the naming convention: <name>/<name>.xml (+ <name>.png).
function conventionModels(sourceDir, source) {
  const models = [];
  for (const name of subdirs(sourceDir)) {
    const dir = `${sourceDir}/${name}`;
    if (!existsSync(`${dir}/${name}.xml`)) { console.warn(`skipping ${dir}: no ${name}.xml`); continue; }
    const screenshot = existsSync(`${dir}/${name}.png`) ? `${name}.png` : undefined;
    models.push({ name, dir, main: `${name}.xml`, group: source, rank: 0, screenshot });
  }
  return models;
}

function indexLibrary() {
  const link = `public/${LIBRARY}`;
  // Without mjcf_assets/ next to package.json, public/mjcf_assets may link
  // to the library elsewhere (on Binder: a volume mounted into the pod,
  // binder/postBuild): index it from public/, where paths are the same.
  const linked = !existsSync(LIBRARY) && existsSync(link);
  if (!existsSync(LIBRARY) && !linked) {
    console.warn(`no ${LIBRARY}/: the asset panel will be empty`);
    writeFileSync('public/assets.json', '[]');
    return;
  }
  if (!existsSync(link) && !lstatSync(link, { throwIfNoEntry: false })) symlinkSync(`../${LIBRARY}`, link);
  if (linked) process.chdir('public');

  const models = [];
  for (const catDir of subdirs(LIBRARY)) {
    const category = CATEGORIES[catDir];
    if (!category) { console.warn(`skipping ${LIBRARY}/${catDir}: not one of ${Object.keys(CATEGORIES)}`); continue; }
    for (const source of subdirs(`${LIBRARY}/${catDir}`)) {
      const sourceDir = `${LIBRARY}/${catDir}/${source}`;
      const found = source === MENAGERIE ? menagerieModels(sourceDir) : conventionModels(sourceDir, source);
      for (const { name, dir, main, group, rank, screenshot } of found) {
        const meta = `${dir}/asset.json`;
        const over = existsSync(meta) ? JSON.parse(readFileSync(meta, 'utf8')) : {};
        const m = { name, dir, main, category, source, group: over.group ?? group };
        if (screenshot) m.screenshot = screenshot;
        m.rank = [FIRST_SOURCES.includes(source) ? FIRST_SOURCES.indexOf(source) : FIRST_SOURCES.length, source,
                  over.group ? -1 : rank, m.group, name];
        models.push(m);
      }
    }
  }
  const cmp = (a, b) => (typeof a === 'string' ? a.localeCompare(b) : a - b);
  models.sort((a, b) => a.rank.reduce((c, v, k) => c || cmp(v, b.rank[k]), 0));
  for (const m of models) delete m.rank;
  if (linked) process.chdir('..');
  writeFileSync('public/assets.json', JSON.stringify(models, null, 1));
  const count = {};
  for (const m of models) count[`${m.category}/${m.source}`] = (count[`${m.category}/${m.source}`] ?? 0) + 1;
  console.log('indexed', models.length, `models of ${LIBRARY}/ into public/assets.json:`, count);
}
indexLibrary();

// Pretrained policies: pretrained_policies/<name>/ (one checkout per model,
// e.g. git clone https://huggingface.co/nepyope/GR00T-WholeBodyControl_g1),
// exposed as public/pretrained_policies (symlink) and listed in
// public/policies.json ({<name>: [its files]}) so the page knows which of
// the policies it registers (POLICIES in public/policy.js) are installed.
const POLICY_DIR = 'pretrained_policies';
function indexPolicies() {
  const stale = lstatSync('public/groot', { throwIfNoEntry: false });  // the old single-policy link
  if (stale?.isSymbolicLink()) unlinkSync('public/groot');
  const index = {};
  if (existsSync(POLICY_DIR)) {
    const link = `public/${POLICY_DIR}`;
    if (!existsSync(link) && !lstatSync(link, { throwIfNoEntry: false })) symlinkSync(`../${POLICY_DIR}`, link);
    for (const name of subdirs(POLICY_DIR)) {
      index[name] = readdirSync(`${POLICY_DIR}/${name}`, { withFileTypes: true })
        .filter((f) => f.isFile() && !f.name.startsWith('.')).map((f) => f.name).sort();
    }
  } else {
    console.warn(`no ${POLICY_DIR}/: no policy can run`);
  }
  writeFileSync('public/policies.json', JSON.stringify(index, null, 1));
  console.log('indexed', Object.keys(index).length, `policies of ${POLICY_DIR}/ into public/policies.json:`, Object.keys(index));
}
indexPolicies();
