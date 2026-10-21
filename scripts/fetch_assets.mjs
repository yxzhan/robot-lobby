// Downloads the GitHub Pages site's model library, the archive pages.json's
// `assets` names (a release asset holding mjcf_assets/<category>/<source>/
// <model>/, packed from part of the library), and unpacks it as
// public/mjcf_assets, where node scripts/vendor.mjs indexes it like the
// library itself. For the build (.github/workflows/pages.yml), on a checkout
// without the library: an existing public/mjcf_assets (a link to the
// library next to package.json, or an earlier download) is left alone.
//
//   node scripts/fetch_assets.mjs [url]
//   node scripts/vendor.mjs && python3 scripts/thumbnails.py
import { execFileSync } from 'node:child_process';
import { existsSync, lstatSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const DEST = 'public/mjcf_assets';
const url = process.argv[2] ?? JSON.parse(readFileSync('pages.json', 'utf8')).assets;
if (!url) throw new Error('no archive: give its URL or set "assets" in pages.json');
if (lstatSync(DEST, { throwIfNoEntry: false })) {
  console.log(`${DEST} is there already, left alone (delete it to download ${url})`);
  process.exit(0);
}

const res = await fetch(url);  // follows GitHub's redirect to its storage
if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
const tmp = mkdtempSync('public/.mjcf_assets-');  // next to DEST: the rename stays on one file system
try {
  const archive = join(tmp, 'mjcf_assets.tar.gz');
  writeFileSync(archive, Buffer.from(await res.arrayBuffer()));
  execFileSync('tar', ['-xzf', archive, '-C', tmp]);  // GNU tar refuses members outside tmp
  if (!existsSync(join(tmp, 'mjcf_assets'))) throw new Error(`${url}: no mjcf_assets/ at the top of the archive`);
  renameSync(join(tmp, 'mjcf_assets'), DEST);
  console.log(`${url} -> ${DEST}`);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
