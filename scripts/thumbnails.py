"""Compresses each asset's screenshot into a small WebP thumbnail for the
asset panel: public/thumbs/<category>/<source>/<model>.webp (thumbPath in
public/js/assets.js), at most WIDTH x HEIGHT, aspect kept. The originals are
not touched.

Reads the library index written by scripts/vendor.mjs (public/json/assets.json,
whose `screenshot` names the image in the model's `dir`), so run that first.
A model the index has no screenshot for still gets one from `<dir>/<name>.png`
(the naming convention) if that image was added after indexing; models with
no image at all are listed. Thumbnails newer than their screenshot are kept.

    python3 scripts/thumbnails.py
"""

import json
import pathlib

from PIL import Image

PUBLIC = pathlib.Path(__file__).resolve().parent.parent / 'public'
WIDTH, HEIGHT = 320, 240  # 2x the panel's card size
QUALITY = 80


def main():
  total_in = total_out = made = 0
  missing = []
  index = PUBLIC / 'json' / 'assets.json'
  if not index.exists():
    raise SystemExit(f'no {index}: run node scripts/vendor.mjs first')
  for model in json.loads(index.read_text()):
    shot = model.get('screenshot')
    if not shot and (PUBLIC / model['dir'] / f"{model['name']}.png").exists():
      shot = f"{model['name']}.png"  # added after the index was written
    if not shot:
      missing.append(f"{model['category']}/{model['source']}/{model['name']}")
      continue
    src = PUBLIC / model['dir'] / shot
    dst = PUBLIC / 'thumbs' / model['category'] / model['source'] / f"{model['name']}.webp"
    if not src.exists():
      print(f'missing {src}')
      continue
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
      dst.parent.mkdir(parents=True, exist_ok=True)
      with Image.open(src) as im:
        im = im.convert('RGBA' if im.mode in ('RGBA', 'LA', 'P') else 'RGB')
        im.thumbnail((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        im.save(dst, 'WEBP', quality=QUALITY, method=6)
      made += 1
    total_in += src.stat().st_size
    total_out += dst.stat().st_size
  print(f'{made} thumbnails written; screenshots {total_in / 2**20:.1f} MB -> '
        f'thumbnails {total_out / 2**20:.2f} MB in {PUBLIC / "thumbs"}')
  if missing:
    print(f'{len(missing)} without a screenshot: {", ".join(missing)}')


if __name__ == '__main__':
  main()
