// Copies the MuJoCo and Filament runtime files out of node_modules so that
// public/ is a self-contained static site (no bundler needed).
import { cpSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, symlinkSync, writeFileSync } from 'node:fs';

const copies = {
  'node_modules/@mujoco/mujoco/mujoco.js': 'public/vendor/mujoco/mujoco.js',
  'node_modules/@mujoco/mujoco/mujoco.wasm': 'public/vendor/mujoco/mujoco.wasm',
  'node_modules/filament/filament.js': 'public/vendor/filament/filament.js',
  'node_modules/filament/filament.wasm': 'public/vendor/filament/filament.wasm',
};
for (const [from, to] of Object.entries(copies)) {
  mkdirSync(to.slice(0, to.lastIndexOf('/')), { recursive: true });
  cpSync(from, to);
}
console.log('vendored', Object.keys(copies).length, 'files into public/vendor');

// If mujoco_menagerie/ is checked out next to package.json, expose it as
// public/menagerie (symlink; http.server follows it) and list each robot's
// MJCF files in public/menagerie.json for the robot picker.
const MENAGERIE = 'mujoco_menagerie';
if (existsSync(MENAGERIE)) {
  const link = 'public/menagerie';
  if (!existsSync(link) && !lstatSync(link, { throwIfNoEntry: false })) symlinkSync(`../${MENAGERIE}`, link);

  // Only robots that ship a scene.xml (robot + floor + lights), one entry each.
  // `main` is the robot file scene.xml includes: spawning that instead of the
  // scene keeps the scene's own floor and lights out of the shared world.
  const robots = [];
  for (const d of readdirSync(MENAGERIE, { withFileTypes: true })) {
    if (!d.isDirectory() || d.name.startsWith('.')) continue;
    const scene = `${MENAGERIE}/${d.name}/scene.xml`;
    if (!existsSync(scene)) continue;
    const main = readFileSync(scene, 'utf8').match(/<include\s+file="([^"]+\.xml)"/)?.[1] ?? 'scene.xml';
    robots.push({ name: d.name, xml: ['scene.xml'], main });
  }
  robots.sort((a, b) => a.name.localeCompare(b.name));
  writeFileSync('public/menagerie.json', JSON.stringify(robots, null, 1));
  console.log('indexed', robots.length, 'menagerie models into public/menagerie.json');
}
