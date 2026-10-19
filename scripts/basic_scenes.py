"""Writes simple test environments and objects into the library
(mjcf_assets/<category>/<source>/<name>/<name>.xml, the naming convention of
scripts/vendor.mjs). Environments, in environments/custom:

  ramp  up a 5 degree slope to a 4 m platform 1 m high, down a 20 degree
        one on the far side; 6 m wide, with low curbs along the edges
  maze  8 x 8 cells of 1.5 m (12 x 12 m) with 2 m walls, entrance in the
        south, exit in the north (green and red pads); a recursive
        backtracker with a fixed seed, so exactly one path between any two
        cells

Both are static geoms centred on the origin, on the floor (z = 0).

Objects, in objects/digital_twin: box_<color> (5 cm cube) and ball_<color>
(5 cm across), red, green and blue, 100 g, on a free joint resting on the
floor, with more friction than MuJoCo's default (sliding 1.5 instead of 1;
torsional on the box, so it does not twist out of a gripper; rolling on the
ball, so it comes to rest).

    python3 scripts/basic_scenes.py [--seed N]
    node scripts/vendor.mjs                 # re-index
    node scripts/render_previews.mjs ...    # screenshots (see its docstring)
"""

import argparse
import math
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'mjcf_assets' / 'environments' / 'custom'
OBJECTS = ROOT / 'mjcf_assets' / 'objects' / 'digital_twin'
H = 2.0  # m: the maze's walls
RAMP_H = 1.0  # m: the ramp's platform


def f(*v):
  return ' '.join(f'{x:.4f}'.rstrip('0').rstrip('.') if isinstance(x, float) else str(x) for x in v)


def write(name, body, out=OUT):
  d = out / name
  d.mkdir(parents=True, exist_ok=True)
  (d / f'{name}.xml').write_text(body)
  print(d / f'{name}.xml')


# ---------------------------------------------------------------------------

def ramp():
  H = RAMP_H
  W, P = 6.0, 4.0                                     # width, platform length
  up, down = math.radians(5), math.radians(20)
  Lu, Ld = H / math.tan(up), H / math.tan(down)
  x0 = -(Lu + P + Ld) / 2                             # foot of the up ramp
  x1, x2, x3 = x0 + Lu, x0 + Lu + P, x0 + Lu + P + Ld
  w = W / 2

  def wedge(name, xa, xb, high_at_b):                 # a solid ramp: its convex hull
    xh = xb if high_at_b else xa
    v = [(xa, -w, 0), (xa, w, 0), (xb, -w, 0), (xb, w, 0), (xh, -w, H), (xh, w, H)]
    return f'<mesh name="{name}" vertex="{f(*[c for p in v for c in p])}"/>'

  CURB_H, CURB_W = 0.12, 0.06

  def curb(xa, za, xb, zb, y):                        # along an edge, from (xa, za) to (xb, zb) on top
    L, a = math.hypot(xb - xa, zb - za), math.atan2(zb - za, xb - xa)
    cx, cz = (xa + xb) / 2, (za + zb) / 2 + CURB_H / 2 * math.cos(a)
    cx -= CURB_H / 2 * math.sin(a)
    q = (math.cos(-a / 2), 0, math.sin(-a / 2), 0)    # about -y: x up towards z
    return (f'<geom type="box" material="curb" size="{f(L / 2, CURB_W / 2, CURB_H / 2)}" '
            f'pos="{f(cx, y, cz)}" quat="{f(*q)}"/>')

  curbs = []
  for y in (-w + CURB_W / 2, w - CURB_W / 2):
    curbs += [curb(x0, 0.0, x1, H, y), curb(x1, H, x2, H, y), curb(x2, H, x3, 0.0, y)]
  write('ramp', f"""<!-- A ramp test course (scripts/basic_scenes.py): up a {math.degrees(up):.0f} degree slope
     ({Lu:.2f} m) to a {P:.0f} m platform {H:.0f} m high, down a {math.degrees(down):.0f} degree one
     ({Ld:.2f} m); {W:.0f} m wide, low curbs along the edges. Centred on the origin. -->
<mujoco model="ramp">
  <asset>
    <texture type="2d" name="deck" builtin="checker" rgb1="0.62 0.64 0.66" rgb2="0.54 0.56 0.58"
             width="256" height="256"/>
    <material name="deck" texture="deck" texrepeat="2 2" texuniform="true"/>
    <material name="up" rgba="0.36 0.56 0.74 1"/>
    <material name="down" rgba="0.80 0.52 0.30 1"/>
    <material name="curb" rgba="0.95 0.80 0.15 1"/>
    {wedge('ramp_up', x0, x1, True)}
    {wedge('ramp_down', x2, x3, False)}
  </asset>
  <worldbody>
    <geom name="ramp_up" type="mesh" mesh="ramp_up" material="up"/>
    <geom name="platform" type="box" material="deck" size="{f(P / 2, w, H / 2)}" pos="{f((x1 + x2) / 2, 0.0, H / 2)}"/>
    <geom name="ramp_down" type="mesh" mesh="ramp_down" material="down"/>
    {(chr(10) + '    ').join(curbs)}
  </worldbody>
</mujoco>
""")


# ---------------------------------------------------------------------------

def maze(seed):
  N, C, T = 8, 1.5, 0.12                              # cells per side, cell size, wall thickness
  rng = random.Random(seed)
  # walls: h[i][j] the wall south of cell (i, j) (row i from the south, i = N the north edge),
  # v[i][j] the wall west of it (j = N the east edge)
  h = [[True] * N for _ in range(N + 1)]
  v = [[True] * (N + 1) for _ in range(N)]
  seen, stack = {(0, 0)}, [(0, 0)]
  while stack:
    i, j = stack[-1]
    nxt = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
           if 0 <= i + di < N and 0 <= j + dj < N and (i + di, j + dj) not in seen]
    if not nxt:
      stack.pop()
      continue
    a, b = rng.choice(nxt)
    if a != i:
      h[max(i, a)][j] = False
    else:
      v[i][max(j, b)] = False
    seen.add((a, b))
    stack.append((a, b))
  start, goal = rng.randrange(N), rng.randrange(N)
  h[0][start] = False                                 # entrance (south)
  h[N][goal] = False                                  # exit (north)

  o = -N * C / 2
  walls = []                                          # merged runs: (x0, y0, x1, y1)
  for i in range(N + 1):                              # east-west runs
    j = 0
    while j < N:
      if not h[i][j]:
        j += 1
        continue
      k = j
      while k < N and h[i][k]:
        k += 1
      walls.append((o + j * C, o + i * C, o + k * C, o + i * C))
      j = k
  for j in range(N + 1):                              # north-south runs
    i = 0
    while i < N:
      if not v[i][j]:
        i += 1
        continue
      k = i
      while k < N and v[k][j]:
        k += 1
      walls.append((o + j * C, o + i * C, o + j * C, o + k * C))
      i = k
  geoms = []
  for n, (xa, ya, xb, yb) in enumerate(walls):
    sx = (xb - xa) / 2 + T / 2 if xb > xa else T / 2  # overlap at the corners: no gaps
    sy = (yb - ya) / 2 + T / 2 if yb > ya else T / 2
    geoms.append(f'<geom name="wall{n}" type="box" material="wall" size="{f(sx, sy, H / 2)}" '
                 f'pos="{f((xa + xb) / 2, (ya + yb) / 2, H / 2)}"/>')
  pad = lambda name, mat, j, y: (f'<geom name="{name}" type="box" material="{mat}" contype="0" conaffinity="0" '
                                 f'size="{f(C / 2 - 0.15, 0.4, 0.003)}" pos="{f(o + (j + 0.5) * C, y, 0.003)}"/>')
  write('maze', f"""<!-- A maze (scripts/basic_scenes.py, seed {seed}): {N} x {N} cells of {C} m, {H:.0f} m walls,
     entrance in the south (green pad), exit in the north (red pad); exactly one path
     between any two cells. Centred on the origin. -->
<mujoco model="maze">
  <asset>
    <texture type="2d" name="wall" builtin="checker" rgb1="0.80 0.76 0.70" rgb2="0.72 0.68 0.62"
             width="256" height="256"/>
    <material name="wall" texture="wall" texrepeat="2 2" texuniform="true"/>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.40 0.44 0.38" rgb2="0.34 0.38 0.32"
             width="256" height="256"/>
    <material name="floor" texture="floor" texrepeat="{f(1 / (2 * C), 1 / (2 * C))}" texuniform="true"/>
    <material name="start" rgba="0.20 0.75 0.30 1"/>
    <material name="goal" rgba="0.85 0.20 0.18 1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="box" material="floor" size="{f(N * C / 2 + 1.0, N * C / 2 + 1.0, 0.005)}" pos="0 0 -0.005"/>
    {pad('start', 'start', start, o - 0.6)}
    {pad('goal', 'goal', goal, -o + 0.6)}
    {(chr(10) + '    ').join(geoms)}
  </worldbody>
</mujoco>
""")


# ---------------------------------------------------------------------------

COLORS = {'red': '0.85 0.18 0.15', 'green': '0.20 0.70 0.25', 'blue': '0.18 0.38 0.85'}
SIZE, MASS = 0.05, 0.1  # m (edge, diameter), kg


def objects():
  r = SIZE / 2
  for color, rgb in COLORS.items():
    for name, geom in [
        ('box', f'type="box" size="{f(r, r, r)}" friction="1.5 0.02 0.0001" condim="4"'),
        ('ball', f'type="sphere" size="{f(r)}" friction="1.5 0.02 0.005" condim="6"'),
    ]:
      write(f'{name}_{color}', f"""<!-- A {color} {name} (scripts/basic_scenes.py): {SIZE * 100:.0f} cm, {MASS * 1000:.0f} g, more friction
     than MuJoCo's default. -->
<mujoco model="{name}_{color}">
  <worldbody>
    <body name="root" pos="0 0 {f(r + 0.002)}">
      <freejoint name="root"/>
      <geom name="{name}" {geom} mass="{f(MASS)}" rgba="{rgb} 1"/>
    </body>
  </worldbody>
</mujoco>
""", OBJECTS)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--seed', type=int, default=7, help='maze layout')
  args = ap.parse_args()
  ramp()
  maze(args.seed)
  objects()


if __name__ == '__main__':
  main()
