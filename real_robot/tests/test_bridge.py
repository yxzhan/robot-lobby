"""End to end with the fake driver: a WebSocket client in the lobby's place
publishes targets, enables, and watches the real robot follow.

    .venv/bin/python -m pytest real_robot/tests   (or run this file)
"""

import asyncio
import json
import threading

from websockets.asyncio.client import connect

from real_robot import drivers
from real_robot.bridge import RealBridge
from real_robot.rosbridge import RosbridgeServer
from real_robot.safety import Supervisor

PORT = 9387
PANDA = drivers.spec('panda')
HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


def test_supervisor_limits():
  s = Supervisor(PANDA, speed=1.0)
  assert 'Sync' in s.enable([q + 0.5 for q in HOME], 0.0, HOME)
  assert s.enable(HOME, 0.0, HOME) == ''
  far = [q + 1.0 for q in HOME]
  step = s.step(far, 0.0, HOME, 0.02)
  assert all(abs(c - h) <= v * 0.02 + 1e-9 for c, h, v in zip(step.q, HOME, PANDA.max_vel))
  assert s.step(far, 1.0, HOME, 0.02).stale  # old target: hold
  for _ in range(40):  # the real robot never moves: a fault
    s.step(far, 0.0, HOME, 0.02)
  assert s.state == 'fault'


def run_bridge():
  server = RosbridgeServer('127.0.0.1', PORT)
  bridge = RealBridge(drivers.driver('panda', 'fake', start=HOME), server, speed=1.0)  # (prefix: panda)
  stop = asyncio.Event()

  async def main():
    t = threading.Thread(target=bridge.run, daemon=True)
    t.start()
    await server.run(stop)
    bridge.stop.set()
  loop = asyncio.new_event_loop()
  th = threading.Thread(target=loop.run_until_complete, args=(main(),), daemon=True)
  th.start()
  return bridge, lambda: loop.call_soon_threadsafe(stop.set)


async def lobby():
  async with connect(f'ws://127.0.0.1:{PORT}') as ws:
    for t in ('panda/real/joint_states', 'panda/real/status'):
      await ws.send(json.dumps({'op': 'subscribe', 'topic': t}))
    names = PANDA.joints + ['finger_joint1', 'finger_joint2']
    real, status = {}, {}

    async def recv_until(pred, timeout=5.0):
      async def go():
        while True:
          m = json.loads(await ws.recv())
          if m['topic'].endswith('status'):
            status.update(json.loads(m['msg']['data']))
          else:
            real.update(zip(m['msg']['name'], m['msg']['position']))
          if pred():
            return
      await asyncio.wait_for(go(), timeout)

    async def target(q, finger=0.04):
      await ws.send(json.dumps({'op': 'publish', 'topic': 'panda/joint_states', 'msg': {'name': names, 'position': q + [finger, finger]}}))

    goal = list(HOME)
    goal[0] = 0.5
    await target(HOME)
    await ws.send(json.dumps({'op': 'publish', 'topic': 'panda/real/enable', 'msg': {'data': True}}))
    await recv_until(lambda: status.get('state') == 'enabled')
    for _ in range(100):  # 2 s of targets at 50 Hz
      await target(goal, 0.0)
      try:
        await recv_until(lambda: abs(real.get('joint1', 0) - 0.5) < 1e-3, 0.02)
        break
      except asyncio.TimeoutError:
        pass
    assert abs(real['joint1'] - 0.5) < 1e-3, real
    await recv_until(lambda: real.get('finger_joint1', 1) < 1e-3)  # the hand closed
    await recv_until(lambda: status.get('stale'), 2.0)  # targets stop: it holds
    await ws.send(json.dumps({'op': 'publish', 'topic': 'panda/real/enable', 'msg': {'data': False}}))
    await recv_until(lambda: status.get('state') == 'disabled')
    return real, status


def test_end_to_end():
  bridge, stop = run_bridge()
  try:
    import time
    time.sleep(0.5)
    real, status = asyncio.run(lobby())
    print('real', real, 'status', status)
  finally:
    stop()


if __name__ == '__main__':
  test_supervisor_limits()
  test_end_to_end()
  print('ok')
