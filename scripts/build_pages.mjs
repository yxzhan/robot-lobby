// Builds the static site for GitHub Pages into dist/: the page (public/,
// with its vendored runtimes), every model of public/mjcf_assets with its
// index and thumbnails, and the policies that pages.json names, with their
// slice of policies.json. In CI public/mjcf_assets is the archive pages.json's
// `assets` names (scripts/fetch_assets.mjs), part of the library; here it is
// usually a link to the whole library, which this would publish. The
// page runs locally on those; its Server panel joins a server
// (server/server.py, e.g. on Binder) for the whole library and a shared
// world (public/js/endpoint.js). .github/workflows/pages.yml runs it on
// every push to main.
//
//   node scripts/fetch_assets.mjs               # public/mjcf_assets, if not there
//   node scripts/vendor.mjs                     # public/vendor, the indexes
//   python3 scripts/thumbnails.py               # mjcf_assets/thumbs
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
import { join } from 'node:path';

const MAX_FILE = 100e6, MAX_SITE = 1e9;

const args = process.argv.slice(2);
const opt = (name, fallback = null) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const out = opt('out', 'dist');
const config = JSON.parse(readFileSync('pages.json', 'utf8'));
const git = (...a) => execFileSync('git', a, { encoding: 'utf8' }).trim();

for (const f of ['public/mjcf_assets', 'public/json/assets.json', 'public/json/policies.json', 'public/vendor']) {
  if (!existsSync(f)) throw new Error(`no ${f}: run node scripts/fetch_assets.mjs and node scripts/vendor.mjs first`);
}
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

// The page: public/ but for the library and its indexes, which follow below.
const LIBRARY = ['mjcf_assets', 'pretrained_policies', 'json/assets.json', 'json/policies.json', 'json/recent.json', 'json/deploy.json'];
cpSync('public', out, {
  recursive: true, dereference: true,
  filter: (src) => !LIBRARY.includes(src.slice('public/'.length)) && !src.split('/').some((p) => p.startsWith('.mjcf_assets-')),
});

// The models: all of public/mjcf_assets (the thumbnails in mjcf_assets/thumbs
// too), its index and the recent list's models that are there.
cpSync('public/mjcf_assets', join(out, 'mjcf_assets'), {
  recursive: true, dereference: true,
  filter: (src) => !src.split('/').includes('.git'),
});
const assets = JSON.parse(readFileSync('public/json/assets.json', 'utf8'));
writeFileSync(join(out, 'json', 'assets.json'), JSON.stringify(assets, null, 1));
const rels = new Set(assets.map((m) => `${m.dir}/${m.main}`));
const recent = JSON.parse(readFileSync('public/json/recent.json', 'utf8')).filter((r) => rels.has(r));
writeFileSync(join(out, 'json', 'recent.json'), JSON.stringify(recent, null, 1));
const thumbs = assets.filter((m) => existsSync(join(out, `mjcf_assets/thumbs/${m.category}/${m.source}/${m.name}.webp`))).length;  // thumbPath

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
if (total > MAX_SITE) console.warn(`over GitHub Pages' ${MAX_SITE / 1e9} GB: pack fewer models into the archive`);

if (args.includes('--push')) {
  const remote = git('remote', 'get-url', opt('push'));
  const run = (...a) => execFileSync('git', ['-C', out, ...a], { stdio: 'inherit' });
  run('init', '-q', '-b', 'gh-pages');
  run('add', '-A');
  run('-c', 'user.name=build_pages', '-c', 'user.email=build_pages@localhost', 'commit', '-q', '-m', `Site from ${git('rev-parse', '--short', 'HEAD')}`);
  run('push', '-f', remote, 'gh-pages');
  rmSync(join(out, '.git'), { recursive: true, force: true });
}
