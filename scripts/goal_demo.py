"""Drives robots of the shared world (server/server.py) back and forth with
goal markers, through its HTTP API: every robot moves `distance` forward,
all wait until each has arrived, then they come back; `times` round trips.
Every browser on the server sees the markers move.

    python3 scripts/goal_demo.py                         # all robots that walk or drive, 1 m, once
    python3 scripts/goal_demo.py --robots r1,r3 --distance 2 --times 3
    python3 scripts/goal_demo.py --url http://localhost:4007 --stop   # clear the goals

The API (see api_goal in server/server.py):
    GET  /api/robots   [{id, name, label, rel, drives, pose {x, y, yaw}, goal, progress, arrived}]
    POST /api/goal     {robots: 'all' | id | 'r3' | label | [...],
                        forward, left (m), turn (rad)  relative to each robot, or
                        goal: {x, y, yaw}              in the world frame, or neither: stop}
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(url, path, body=None):
  req = urllib.request.Request(url.rstrip('/') + path, method='POST' if body is not None else 'GET',
                               data=None if body is None else json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json'})
  try:
    with urllib.request.urlopen(req, timeout=10) as res:
      return json.loads(res.read())
  except urllib.error.HTTPError as err:
    sys.exit(f'{path}: {err.code} {err.read().decode()}')


def wait(url, names, timeout):
  """Until none of `names` has a goal left; False if one gave up (cleared) or time ran out."""
  t0 = time.monotonic()
  while time.monotonic() - t0 < timeout:
    robots = [r for r in call(url, '/api/robots') if r['name'] in names]
    if all(r['goal'] is None for r in robots):
      return all(r['arrived'] for r in robots)
    time.sleep(0.2)
  return False


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--url', default='http://localhost:4007')
  ap.add_argument('--robots', default='all', help="'all', or names / labels separated by commas (r1,r3)")
  ap.add_argument('--distance', type=float, default=1.0, help='m forward, then back')
  ap.add_argument('--times', type=int, default=1, help='round trips')
  ap.add_argument('--timeout', type=float, default=60, help='s for each leg')
  ap.add_argument('--stop', action='store_true', help='clear their goals and exit')
  args = ap.parse_args()
  which = 'all' if args.robots == 'all' else args.robots.split(',')
  if args.stop:
    call(args.url, '/api/goal', {'robots': which})
    return
  for trip in range(args.times):
    for leg, d in (('forward', args.distance), ('back', -args.distance)):
      robots = call(args.url, '/api/goal', {'robots': which, 'forward': d})
      idle = [r['name'] for r in robots if not r['drives']]
      if idle:
        print(f"  {', '.join(idle)}: no walking policy or base switched on, they will not move")
      names = {r['name'] for r in robots if r['drives']}
      if not names:
        sys.exit('no robot to drive')
      t0 = time.monotonic()
      ok = wait(args.url, names, args.timeout)
      poses = {r['name']: r['pose'] for r in call(args.url, '/api/robots')}
      at = ', '.join(f"{n} ({poses[n]['x']:.2f}, {poses[n]['y']:.2f})" for n in sorted(names))
      print(f"trip {trip + 1} {leg:7} {'arrived' if ok else 'NOT all arrived'} in {time.monotonic() - t0:.1f} s: {at}")
      if not ok:
        call(args.url, '/api/goal', {'robots': sorted(names)})  # stop them
        sys.exit(1)


if __name__ == '__main__':
  main()
