// Builds the static site for GitHub Pages into dist/: the page (public/,
// with its vendored runtimes) and the part of the model library and the
// policies that pages.json names, with their slice of assets.json,
// policies.json and the thumbnails. The page runs locally on those; its
// Server panel joins a server (server/server.py, e.g. on Binder) for the
// whole library and a shared world (public/js/endpoint.js).
//
//   node scripts/vendor.mjs && python3 scripts/thumbnails.py   # public/ up to date
//   node scripts/build_pages.mjs [--out dist] [--binder gh/<owner>/<repo>/<ref>] [--hub https://mybinder.org]
//   node scripts/build_pages.mjs --push origin    # and publish it as branch gh-pages
//
// --push makes dist/ a one-commit repository and force-pushes it to the
// gh-pages branch of that remote, so the branch never grows a history of
// model files (set the repository's Pages source to that branch). GitHub
// Pages takes files up to 100 MB and sites up to about 1 GB: both are
// checked.
//
// json/deploy.json tells the Server panel which Binder repository to start:
// pages.json's binder, its repo by default this checkout's GitHub remote and
// branch.
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

const MAX_FILE = 100e6, MAX_SITE = 1e9;

const args = process.argv.slice(2);
const opt = (name, fallback = null) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const out = opt('out', 'dist');
const config = JSON.parse(readFileSync('pages.json', 'utf8'));
const git = (...a) => execFileSync('git', a, { encoding: 'utf8' }).trim();

for (const f of ['public/json/assets.json', 'public/json/policies.json', 'public/vendor']) {
  if (!existsSync(f)) throw new Error(`no ${f}: run node scripts/vendor.mjs first`);
}
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

// The page: public/ but for the library, its indexes and the thumbnails,
// which follow in their pages.json slice.
const LIBRARY = ['mjcf_assets', 'pretrained_policies', 'thumbs', 'json/assets.json', 'json/policies.json', 'json/deploy.json'];
cpSync('public', out, {
  recursive: true, dereference: true,
  filter: (src) => !LIBRARY.includes(src.slice('public/'.length)),
});

// The models: whole directories (a menagerie directory holds several), but
// not the screenshots thumbnails.py made the thumbnails from.
const picked = config.models.map((d) => `mjcf_assets/${d.replace(/\/$/, '')}`);
const isPicked = (dir) => picked.some((p) => dir === p || dir.startsWith(`${p}/`));
const assets = JSON.parse(readFileSync('public/json/assets.json', 'utf8')).filter((m) => isPicked(m.dir));
for (const p of picked) if (!existsSync(p)) throw new Error(`pages.json: no ${p}`);
const screenshots = new Set(assets.filter((m) => m.screenshot).map((m) => join(m.dir, m.screenshot)));
for (const p of picked) {
  cpSync(p, join(out, p), {
    recursive: true, dereference: true,
    filter: (src) => !src.split('/').includes('.git') && !screenshots.has(src),
  });
}
// Directories a picked model takes its meshes and textures from through
// <compiler meshdir/texturedir> outside its own (objects/robocasa_spawn ->
// objects/robocasa, scripts/import_robocasa.py).
const outside = new Set();
for (const m of assets) {
  const compiler = readFileSync(join(m.dir, m.main), 'utf8').match(/<compiler\b[^>]*>/)?.[0] ?? '';
  for (const [, d] of compiler.matchAll(/\b(?:meshdir|texturedir|assetdir)="([^"]+)"/g)) {
    const p = join(m.dir, d);
    if (!isPicked(p)) outside.add(p);
  }
}
for (const p of outside) cpSync(p, join(out, p), { recursive: true, dereference: true });
writeFileSync(join(out, 'json', 'assets.json'), JSON.stringify(assets, null, 1));
let thumbs = 0;
for (const m of assets) {
  const t = `thumbs/${m.category}/${m.source}/${m.name}.webp`;  // thumbPath in public/js/assets.js
  if (!existsSync(`public/${t}`)) continue;
  mkdirSync(dirname(join(out, t)), { recursive: true });
  cpSync(`public/${t}`, join(out, t));
  thumbs++;
}

// The policies.
const installed = JSON.parse(readFileSync('public/json/policies.json', 'utf8'));
const policies = Object.fromEntries(config.policies.filter((p) => {
  if (!installed[p]) console.warn(`pages.json: policy ${p} is not installed, left out`);
  return installed[p];
}).map((p) => [p, installed[p]]));
for (const [p, files] of Object.entries(policies)) {
  for (const f of files) {
    mkdirSync(join(out, 'pretrained_policies', p), { recursive: true });
    cpSync(join('pretrained_policies', p, f), join(out, 'pretrained_policies', p, f), { dereference: true });
  }
}
writeFileSync(join(out, 'json', 'policies.json'), JSON.stringify(policies, null, 1));

// The Binder repository for the Server panel: gh/<owner>/<repo>/<branch>.
function githubRepo() {
  try {
    const url = git('remote', 'get-url', args.includes('--push') ? opt('push') : 'origin');
    const m = url.match(/github\.com[:/]([^/]+)\/(.+?)(\.git)?$/);
    return m && `gh/${m[1]}/${m[2]}/${git('rev-parse', '--abbrev-ref', 'HEAD')}`;
  } catch {
    return null;
  }
}
const binder = { ...config.binder, repo: opt('binder') ?? config.binder?.repo ?? githubRepo(), hub: opt('hub') ?? config.binder?.hub };
if (binder.repo) writeFileSync(join(out, 'json', 'deploy.json'), JSON.stringify({ binder }, null, 1));
else console.warn('no Binder repository (no GitHub remote, no --binder): the Server panel can only join a server by its URL');
writeFileSync(join(out, '.nojekyll'), '');  // serve files as they are

// Pages' limits.
let total = 0;
const big = [];
(function walk(dir) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) { walk(p); continue; }
    const size = statSync(p).size;
    total += size;
    if (size > MAX_FILE) big.push(`${p} (${(size / 1e6).toFixed(0)} MB)`);
  }
})(out);
console.log(`${out}/: ${assets.length} models (${thumbs} thumbnails), ${Object.keys(policies).length} policies, ` +
  `${(total / 1e6).toFixed(0)} MB${binder.repo ? `; Binder: ${binder.hub} ${binder.repo}` : ''}`);
if (big.length) throw new Error(`over GitHub's ${MAX_FILE / 1e6} MB per file: ${big.join(', ')}`);
if (total > MAX_SITE) console.warn(`over GitHub Pages' ${MAX_SITE / 1e9} GB: take models out of pages.json`);

if (args.includes('--push')) {
  const remote = git('remote', 'get-url', opt('push'));
  const run = (...a) => execFileSync('git', ['-C', out, ...a], { stdio: 'inherit' });
  run('init', '-q', '-b', 'gh-pages');
  run('add', '-A');
  run('-c', 'user.name=build_pages', '-c', 'user.email=build_pages@localhost', 'commit', '-q', '-m', `Site from ${git('rev-parse', '--short', 'HEAD')}`);
  run('push', '-f', remote, 'gh-pages');
  rmSync(join(out, '.git'), { recursive: true, force: true });
}
