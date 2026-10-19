"""Packs the parts of the local model library and policies that the Binder
image cannot fetch or generate itself (binder/postBuild clones
mujoco_menagerie and runs scripts/basic_scenes.py) into one .tar.gz per
source, for uploading somewhere public (e.g. assets of a GitHub release,
2 GB each at most), and prints the lines for binder/library.txt.

    python3 scripts/pack_library.py [--out library_packs] [--url https://github.com/<owner>/<repo>/releases/download/<tag>/]
    python3 scripts/pack_library.py --list    # what it would pack

Packed: every mjcf_assets/<category>/<source>/ but mujoco_menagerie, without
the models basic_scenes.py writes and without git metadata; every
pretrained_policies/<dir>/ that binder/library.txt does not download already.
"""

import argparse
import pathlib
import re
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATED = {'environments/custom/ramp', 'environments/custom/maze'} | {
  f'objects/digital_twin/{k}_{c}' for k in ('box', 'ball') for c in ('red', 'green', 'blue')}
LIMIT = 2e9


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--out', default='library_packs')
  ap.add_argument('--url', default='<url of the uploaded files>/')
  ap.add_argument('--list', action='store_true', help='only list the archives and what goes in them')
  args = ap.parse_args()
  out = ROOT / args.out
  if not args.list:
    out.mkdir(exist_ok=True)
  listed = (ROOT / 'binder' / 'library.txt').read_text()
  packs = []  # (archive, directory it unpacks into, members relative to it)
  lib = ROOT / 'mjcf_assets'
  for src in sorted(p for p in lib.glob('*/*') if p.is_dir()):
    if src.name == 'mujoco_menagerie':
      continue
    rel = src.relative_to(lib).as_posix()
    models = [m for m in sorted(src.iterdir()) if f'{rel}/{m.name}' not in GENERATED]
    if models:
      packs.append((f"mjcf_assets-{rel.replace('/', '-')}.tar.gz", f'mjcf_assets/{rel}', models))
  for pol in sorted(p for p in (ROOT / 'pretrained_policies').iterdir() if p.is_dir()):
    if f'pretrained_policies/{pol.name}' not in listed:
      packs.append((f'pretrained_policies-{pol.name}.tar.gz', f'pretrained_policies/{pol.name}', sorted(pol.iterdir())))
  lines = []
  for name, dest, members in packs:
    lines.append(f'{args.url}{name} {dest}')
    if args.list:
      print(f"{name}: {', '.join(m.name for m in members)}")
      continue
    path = out / name
    with tarfile.open(path, 'w:gz', dereference=True) as tar:
      for m in members:
        tar.add(m, arcname=m.name, filter=lambda t: None if re.search(r'(^|/)\.git(/|$)', t.name) else t)
    size = path.stat().st_size
    print(f'{path.relative_to(ROOT)}: {size / 1e6:.0f} MB{"  OVER 2 GB" if size > LIMIT else ""}')
  print('\nfor binder/library.txt:')
  print('\n'.join(lines))


if __name__ == '__main__':
  main()
