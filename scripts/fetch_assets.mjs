// Gets the GitHub Pages site's model library, the branch pages.json's
// `assets` names in the library repository ({repo, branch}: the `lite`
// branch, part of the library laid out as <category>/<source>/<model>/),
// as public/mjcf_assets, where node scripts/vendor.mjs indexes it like the
// library itself: a shallow clone, without its .git. For the build
// (.github/workflows/pages.yml), on a checkout without the library: an
// existing public/mjcf_assets (a link to the library next to package.json,
// or an earlier fetch) is left alone.
//
//   node scripts/fetch_assets.mjs [repo [branch]]
//   node scripts/vendor.mjs && python3 scripts/thumbnails.py
import { execFileSync } from 'node:child_process';
import { lstatSync, mkdtempSync, readFileSync, renameSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const DEST = 'public/mjcf_assets';
const config = JSON.parse(readFileSync('pages.json', 'utf8')).assets ?? {};
const repo = process.argv[2] ?? config.repo, branch = process.argv[3] ?? config.branch ?? 'lite';
if (!repo) throw new Error('no library repository: give its URL or set "assets": {"repo"} in pages.json');
if (lstatSync(DEST, { throwIfNoEntry: false })) {
  console.log(`${DEST} is there already, left alone (delete it to fetch ${repo} ${branch})`);
  process.exit(0);
}

const tmp = mkdtempSync('public/.mjcf_assets-');  // next to DEST: the rename stays on one file system
try {
  const out = join(tmp, 'mjcf_assets');
  execFileSync('git', ['clone', '--quiet', '--depth', '1', '--single-branch', '--branch', branch, repo, out], { stdio: 'inherit' });
  rmSync(join(out, '.git'), { recursive: true, force: true });  // the files only
  renameSync(out, DEST);
  console.log(`${repo} ${branch} -> ${DEST}`);
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
