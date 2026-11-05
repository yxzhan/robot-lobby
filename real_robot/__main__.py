"""python -m real_robot --robot panda [--ip 172.16.0.2] [--driver fake] ...

Runs a rosbridge-compatible WebSocket for the lobby and a real robot that
follows the lobby's robot (bridge.py). In the lobby: the Bridge address
<this machine>:9090 in the ROS panel, the robot's ROS on, then Real robot ›
Sync and Enable (see real_robot/README.md)."""

import argparse
import asyncio
import logging
import signal
import threading

from . import drivers
from .bridge import RealBridge
from .rosbridge import RosbridgeServer


def main():
  p = argparse.ArgumentParser(prog='python -m real_robot', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--robot', default='panda', choices=sorted(drivers.ROBOTS))
  p.add_argument('--driver', default='real', choices=['real', 'fake'], help='fake: a simulated stand-in (no hardware)')
  p.add_argument('--host', default='0.0.0.0', help='address the WebSocket listens on')
  p.add_argument('--port', type=int, default=9090)
  p.add_argument('--prefix', default='r1', help="the lobby robot's ROS prefix (its id, r1 for the first spawned)")
  p.add_argument('--rate', type=float, default=50.0, help='control loop, Hz')
  p.add_argument('--speed', type=float, default=0.5, help='fraction of the joints\' max speeds the bridge commands at most')
  p.add_argument('--timeout', type=float, default=0.3, help='s without targets before the robot holds')
  p.add_argument('--ip', default='172.16.0.2', help='Panda: the robot\'s address (FCI)')
  p.add_argument('--dynamics', type=float, default=0.15, help='Panda: franky\'s relative_dynamics_factor')
  p.add_argument('--grasp-force', type=float, default=20.0, help='Panda: N')
  p.add_argument('--serial', default='/dev/ttyACM0', help='SO-101: the servo bus')
  p.add_argument('--calibration', help='SO-101: calibration file')
  p.add_argument('--lag', type=float, default=0.0, help='fake: s its states lag behind')
  a = p.parse_args()
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%H:%M:%S')

  options = {'fake': {'lag': a.lag},
             'panda': {'ip': a.ip, 'dynamics': a.dynamics, 'grasp_force': a.grasp_force},
             'so101': {'port': a.serial, 'calibration': a.calibration}}
  opts = options['fake'] if a.driver == 'fake' else options.get(a.robot, {})
  server = RosbridgeServer(a.host, a.port)
  bridge = RealBridge(drivers.driver(a.robot, a.driver, **opts), server, ns=a.prefix,
                      rate=a.rate, speed=a.speed, timeout=a.timeout)

  async def serve():
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
      loop.add_signal_handler(sig, stop.set)
    control = threading.Thread(target=bridge.run, name='control')
    control.start()
    watch = asyncio.ensure_future(asyncio.to_thread(control.join))  # the control thread died: exit
    run = asyncio.ensure_future(server.run(stop))
    await asyncio.wait([watch, run, asyncio.ensure_future(stop.wait())], return_when=asyncio.FIRST_COMPLETED)
    bridge.stop.set()
    stop.set()
    await asyncio.to_thread(control.join)
    run.cancel()

  asyncio.run(serve())


main()
