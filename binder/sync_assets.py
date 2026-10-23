"""Copies the local model library (mjcf_assets/) into a Binder session's,
the shared volume its public/mjcf_assets links to, sending only what is new
or changed.

    from binder_session import BinderTerminal
    from sync_assets import sync_assets

    term = BinderTerminal(result["url"], result["token"])
    plan = sync_assets(term)                # dry run: lists what would change, writes nothing
    sync_assets(term, dry_run=False)        # copies it, then indexes the library again

1. The session lists its library (path, size, mtime) through the terminal,
   which is compared with the local one: files missing there or of another
   size or mtime are sent, except those of the same size whose md5 is the
   same on both sides (only the mtime differs: copied at another time). One
   that is newer there was changed by someone
   else and is left alone unless overwrite_newer=True; files only there
   stay unless delete=True (the volume is shared; emptied directories stay
   too).
2. They go in parts of at most `part_size` bytes: each is packed into a tar
   (zstd-compressed if the session has zstd; tar keeps the mtimes, so the
   next comparison holds), uploaded into the session's Jupyter directory
   through the Contents API in chunks, as JupyterLab uploads a large file
   (PUT {url}api/contents/<path>, {type: file, format: base64, content,
   chunk: 1, 2, …, -1 for the last}), unpacked into the library through
   the terminal and deleted. An interrupted sync just runs again: what was
   unpacked already compares equal.
3. The session's library is indexed again (scripts/vendor.mjs,
   scripts/thumbnails.py, as binder/start does at startup).

Left out on both sides: .git directories and saved_stage/ (each server's
own saved stages). The thumbnails (mjcf_assets/thumbs) go like the models.
"""

import base64
import gzip
import hashlib
import os
import pathlib
import secrets
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCAL = ROOT / 'mjcf_assets'
# The session's library: where its repo's public/mjcf_assets points.
REMOTE = '"$(readlink -f "${REPO_DIR:-$HOME}/public/mjcf_assets")"'
SKIP_DIRS = {'.git'}         # at any depth
SKIP_TOP = {'saved_stage'}   # directly under the library
CHUNK = 8 << 20              # bytes per upload request (JupyterLab sends 1 MB)
PART = 512 << 20             # bytes of files per uploaded tar


def skipped(rel):
  parts = rel.split('/')
  return parts[0] in SKIP_TOP or any(p in SKIP_DIRS for p in parts)


def local_files(root=None):
  """{relative path: (size, whole-second mtime)} of the local library (LOCAL)."""
  root = root or LOCAL
  out = {}
  for d, dirs, files in os.walk(root):
    rel_dir = os.path.relpath(d, root)
    dirs[:] = [x for x in dirs if not skipped(os.path.normpath(os.path.join(rel_dir, x)))]
    for f in files:
      rel = os.path.normpath(os.path.join(rel_dir, f))
      if not skipped(rel):
        st = os.stat(os.path.join(d, f))
        out[rel] = (st.st_size, int(st.st_mtime))
  return out


def run(term, cmd, timeout=None, stream=False, check=True):
  out, rc = term.run(cmd, timeout=timeout, stream=stream)
  if check and rc:
    raise RuntimeError(f'in the session, exit {rc}: {cmd[:200]}\n{out[-2000:]}')
  return out


def upload(term, path, name, chunk=CHUNK, progress=None):
  """Uploads local file `path` as `name` in the session's Jupyter directory,
  in chunks; checks the size it arrived with."""
  url = term.url + 'api/contents/' + quote(name)
  size, done, start = os.path.getsize(path), 0, time.monotonic()
  with open(path, 'rb') as f:
    data, i = f.read(chunk), 1
    while True:
      nxt = f.read(chunk)
      model = {'type': 'file', 'format': 'base64', 'name': name.split('/')[-1], 'path': name,
               'content': base64.b64encode(data).decode()}
      if i > 1 or nxt:  # one chunk: a plain save
        model['chunk'] = i if nxt else -1
      r = requests.put(url, json=model, headers=term.headers, timeout=600)
      if not r.ok:
        raise RuntimeError(f'upload of {name}: HTTP {r.status_code} {r.text[:300]}')
      done += len(data)
      if progress:
        progress(done, size, time.monotonic() - start)
      if not nxt:
        break
      data, i = nxt, i + 1
  got = requests.get(url, params={'content': 0}, headers=term.headers, timeout=60).json().get('size')
  if got != size:
    raise RuntimeError(f'{name} arrived with {got} bytes, not {size}')


def remote_info(term):
  """The session's library: its path, its files (as local_files), where
  the Jupyter directory (the uploads') is, its free space, and whether it
  has zstd."""
  # Where uploads land: a probe file found by the terminal.
  probe = f'mjcf-sync-probe-{secrets.token_hex(6)}'
  with tempfile.NamedTemporaryFile('wb', delete=False) as f:
    f.write(b'probe')
  try:
    upload(term, f.name, probe)
  finally:
    os.unlink(f.name)
  head = run(term, f'D={REMOTE}; echo "$D"; test -d "$D" && test -w "$D" && echo writable; '
                   'command -v zstd >/dev/null && echo has-zstd; '
                   f'for j in "${{JUPYTER_SERVER_ROOT:-}}" "${{REPO_DIR:-$HOME}}" "$HOME" "$PWD"; do '
                   f'test -n "$j" && test -f "$j/{probe}" && {{ echo "jupyter $j"; rm "$j/{probe}"; '
                   'echo "free $(df -B1 --output=avail "$j" | tail -1)"; break; }; done', timeout=60)
  lines = head.splitlines()
  if 'writable' not in lines:
    raise RuntimeError(f'the session has no writable library at {lines[0] if lines else "?"}: {head}')
  jup = next((l[8:] for l in lines if l.startswith('jupyter ')), None)
  if not jup:
    raise RuntimeError(f'could not find where the session keeps uploads (look for {probe} there): {head}')
  # The listing, gzipped and in base64 on one line (the terminal mangles nothing in it).
  blob = run(term, f'cd {REMOTE} && find . -type f -printf "%P\\t%s\\t%T@\\n" | gzip -c | base64 -w0; echo', timeout=600)
  files = {}
  for row in gzip.decompress(base64.b64decode(blob.strip())).decode().splitlines():
    rel, size, mtime = row.split('\t')
    if not skipped(rel):
      files[rel] = (int(size), int(float(mtime)))
  free = next((int(l[5:]) for l in lines if l.startswith('free ')), None)
  return {'dir': lines[0], 'files': files, 'jupyter': jup, 'free': free, 'zstd': 'has-zstd' in lines}


def same_content(term, info, paths):
  """Those of `paths` (relative to both libraries) whose md5 is the same here and there."""
  if not paths:
    return set()
  name = f'mjcf-sync-md5-{secrets.token_hex(6)}'
  with tempfile.NamedTemporaryFile('wb', delete=False) as f:
    f.write(b'\0'.join(p.encode() for p in paths) + b'\0')
  try:
    upload(term, f.name, name)
  finally:
    os.unlink(f.name)
  blob = run(term, f'F="{info["jupyter"]}/{name}"; cd {REMOTE} && xargs -0 md5sum -- < "$F" | gzip -c | base64 -w0; '
                   'rc=$?; rm -f "$F"; echo; (exit $rc)', timeout=1800)
  there = {}
  for row in gzip.decompress(base64.b64decode(blob.strip())).decode().splitlines():
    digest, rel = row.split('  ', 1)
    there[rel] = digest
  here = {}
  for p in paths:
    h = hashlib.md5()
    with open(LOCAL / p, 'rb') as f:
      while chunk := f.read(1 << 20):
        h.update(chunk)
    here[p] = h.hexdigest()
  return {p for p in paths if there.get(p) == here[p]}


def plan_sync(local, remote):
  """What a sync does: {new, changed, remote_newer, remote_only}."""
  new = sorted(k for k in local if k not in remote)
  changed, newer = [], []
  for k in sorted(local.keys() & remote.keys()):
    if local[k] != remote[k]:
      (newer if remote[k][1] > local[k][1] else changed).append(k)
  return {'new': new, 'changed': changed, 'remote_newer': newer,
          'remote_only': sorted(k for k in remote if k not in local)}


def summary(paths, depth=3, top=12):
  """Counts per directory (the first `depth` parts), largest first."""
  groups = {}
  for p in paths:
    g = '/'.join(p.split('/')[:depth])
    groups[g] = groups.get(g, 0) + 1
  rows = sorted(groups.items(), key=lambda kv: -kv[1])
  lines = [f'    {n:6d}  {g}' for g, n in rows[:top]]
  if len(rows) > top:
    lines.append(f'    {sum(n for _, n in rows[top:]):6d}  ({len(rows) - top} more directories)')
  return '\n'.join(lines)


def human(n):
  for unit in ('B', 'KB', 'MB', 'GB'):
    if n < 1024 or unit == 'GB':
      return f'{n} B' if unit == 'B' else f'{n:.1f} {unit}'
    n /= 1024


def parts(paths, sizes, limit):
  """`paths` in groups of at most `limit` bytes (a larger file alone)."""
  group, total = [], 0
  for p in paths:
    if group and total + sizes[p] > limit:
      yield group
      group, total = [], 0
    group.append(p)
    total += sizes[p]
  if group:
    yield group


def pack(paths, out, zstd):
  """A tar of `paths` (relative to LOCAL) at `out`."""
  with tempfile.NamedTemporaryFile('wb', delete=False) as f:
    f.write(b'\0'.join(p.encode() for p in paths) + b'\0')
    listing = f.name
  try:
    cmd = ['tar', '-C', str(LOCAL), '--null', '-T', listing, '-cf', str(out)]
    if zstd:
      cmd[1:1] = ['--use-compress-program', 'zstd -T0 -3']
    subprocess.run(cmd, check=True)
  finally:
    os.unlink(listing)


def sync_assets(term, dry_run=True, delete=False, overwrite_newer=False, reindex=True, part_size=PART):
  """Brings the session's library up to the local one (see the top).
  Returns the plan: {dir, new, changed, remote_newer, remote_only,
  same_content, bytes}."""
  print('listing the local library…', flush=True)
  local = local_files()
  print('listing the session\'s library…', flush=True)
  info = remote_info(term)
  plan = plan_sync(local, info['files'])
  maybe = [k for k in plan['changed'] + plan['remote_newer'] if local[k][0] == info['files'][k][0]]
  if maybe:
    print(f'comparing the content of {len(maybe)} files of the same size…', flush=True)
  same = same_content(term, info, maybe)
  for key in ('changed', 'remote_newer'):
    plan[key] = [k for k in plan[key] if k not in same]
  plan['same_content'] = sorted(same)
  send = plan['new'] + plan['changed'] + (plan['remote_newer'] if overwrite_newer else [])
  plan.update(dir=info['dir'], bytes=sum(local[k][0] for k in send))
  print(f"session library: {info['dir']} ({len(info['files'])} files); local: {len(local)} files")
  for key, text in (('same_content', 'differ in mtime only (same content), left as they are'),
                    ('new', 'new here, to send'), ('changed', 'changed here, to send'),
                    ('remote_newer', 'newer there, ' + ('to overwrite' if overwrite_newer else 'kept (overwrite_newer=True sends them)')),
                    ('remote_only', 'only there, ' + ('to delete' if delete else 'kept (delete=True deletes them)'))):
    if plan[key]:
      print(f'  {len(plan[key])} {text}:\n{summary(plan[key])}')
  print(f'to send: {len(send)} files, {human(plan["bytes"])}; uploads go through {info["jupyter"]}'
        + (f' ({human(info["free"])} free)' if info['free'] is not None else ''))
  if dry_run:
    print('dry run: nothing written (sync_assets(term, dry_run=False) copies it)')
    return plan
  if not send and not (delete and plan['remote_only']):
    print('nothing to do')
    return plan
  if info['free'] is not None and info['free'] < 2 * part_size:
    raise RuntimeError(f"only {human(info['free'])} free in {info['jupyter']}: pass a smaller part_size")

  zstd = info['zstd'] and shutil.which('zstd') is not None
  ext = '.tar.zst' if zstd else '.tar'
  unpack = 'zstd -dc "$F" | tar -x' if zstd else 'tar -x -f "$F"'
  tmp = pathlib.Path(tempfile.mkdtemp(prefix='mjcf-sync-'))
  groups = list(parts(send, {k: local[k][0] for k in send}, part_size))
  sent = 0
  try:
    for n, group in enumerate(groups, 1):
      size = sum(local[k][0] for k in group)
      name = f'mjcf-sync-{secrets.token_hex(6)}{ext}'
      print(f'part {n}/{len(groups)}: {len(group)} files, {human(size)}: packing…', flush=True)
      pack(group, tmp / name, zstd)
      shown = [0]

      def progress(done, total, secs):
        if done == total or done - shown[0] >= total / 4:
          shown[0] = done
          print(f'  uploaded {human(done)} of {human(total)} ({human(done / max(secs, 1e-3))}/s)', flush=True)
      try:
        upload(term, tmp / name, name, progress=progress)
        run(term, f'( set -o pipefail; F="{info["jupyter"]}/{name}"; {unpack} -C {REMOTE} --no-same-owner )')
      finally:
        run(term, f'rm -f "{info["jupyter"]}/{name}"', timeout=60, check=False)
        (tmp / name).unlink(missing_ok=True)
      sent += len(group)
      print(f'  unpacked ({sent}/{len(send)} files)', flush=True)

    if delete and plan['remote_only']:
      print(f"deleting {len(plan['remote_only'])} files there…", flush=True)
      name = f'mjcf-sync-delete-{secrets.token_hex(6)}'
      (tmp / name).write_bytes(b'\0'.join(p.encode() for p in plan['remote_only']) + b'\0')
      upload(term, tmp / name, name)
      run(term, f'F="{info["jupyter"]}/{name}"; cd {REMOTE} && xargs -0 rm -f -- < "$F"; rc=$?; rm -f "$F"; (exit $rc)',
          timeout=600)
  finally:
    shutil.rmtree(tmp, ignore_errors=True)

  if reindex:
    print('indexing the session\'s library…', flush=True)
    print(run(term, 'cd "${REPO_DIR:-$HOME}" && node scripts/vendor.mjs > /tmp/index.log 2>&1 '
                    '&& python3 scripts/thumbnails.py >> /tmp/index.log 2>&1; rc=$?; tail -n 3 /tmp/index.log; (exit $rc)',
              check=False))
  print('done')
  return plan
