"""The lobby's robot as the target of a real one. Topics, under the lobby
robot's prefix (`ns`: the robot's name, as its ROS profile sets it, e.g.
panda; changed in the ROS panel for a second one):

  <ns>/joint_states       lobby -> here   the lobby robot's joints: the target
  <ns>/real/joint_states  here -> lobby   the real robot's (sensor_msgs/JointState,
                                          the lobby's joint names): its ghost
  <ns>/real/status        here -> lobby   std_msgs/String, JSON: {robot, state
                                          (disabled | enabled | fault), reason,
                                          stale, error (rad), clients}
  <ns>/real/enable        lobby -> here   std_msgs/Bool: enable (true) / disable

The control loop runs on a thread of its own at `rate` Hz: it reads the
robot, runs the safety layer (safety.py) on the latest target and sends the
command; the WebSocket's thread only stores what comes in.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from .safety import ENABLED, Supervisor

log = logging.getLogger('bridge')

STATUS_S = 0.2  # status at least this often


class RealBridge:
  def __init__(self, driver, server, ns=None, rate=50.0, state_rate=30.0, speed=0.5, timeout=0.3):
    self.driver, self.spec, self.server = driver, driver.spec, server
    self.ns = (self.spec.name if ns is None else ns).strip('/')
    self.rate, self.state_rate = rate, state_rate
    self.safety = Supervisor(self.spec, timeout=timeout, speed=speed)
    self.lock = threading.Lock()
    self.target = None        # arm joint positions, spec.joints' order
    self.target_grip = None   # 0 open .. 1 closed
    self.target_t = 0.0       # time.monotonic() it came
    self.requests = []        # enable (True) / disable (False), in order
    self.clients = 0
    self.real = None          # the last RobotState
    self.stop = threading.Event()
    self.status = None
    self.topics = {k: self.topic(t) for k, t in
                   (('target', 'joint_states'), ('real', 'real/joint_states'), ('status', 'real/status'), ('enable', 'real/enable'))}
    server.on(self.topics['target'], self.on_target)
    server.on(self.topics['enable'], self.on_enable)
    server.on_clients = self.on_clients

  def topic(self, name):
    return f'{self.ns}/{name}' if self.ns else name

  # ---- from the WebSocket's thread ----

  def on_target(self, msg):
    names, pos = msg.get('name') or [], msg.get('position') or []
    by = {n: p for n, p in zip(names, pos) if isinstance(p, (int, float))}
    if not all(j in by for j in self.spec.joints):
      return  # not this robot's joints (a wrong prefix?)
    g = self.spec.gripper
    with self.lock:
      self.target = [float(by[j]) for j in self.spec.joints]
      if g and g.joint in by:
        self.target_grip = g.fraction(by[g.joint])
      self.target_t = time.monotonic()

  def on_enable(self, msg):
    with self.lock:
      self.requests.append(bool(msg.get('data')))

  def on_clients(self, n):
    self.clients = n
    if not n:  # the last lobby left: nobody watches the robot
      with self.lock:
        self.requests.append(False)

  # ---- the control thread ----

  def run(self):
    self.driver.connect()
    log.info('%s connected; topics %s', self.spec.name, ', '.join(self.topics.values()))
    dt = 1.0 / self.rate
    next_t = time.monotonic()
    last_state = last_status = 0.0
    holding = True
    try:
      while not self.stop.is_set():
        now = time.monotonic()
        real = self.real = self.driver.read()
        with self.lock:
          target, grip, age = self.target, self.target_grip, now - self.target_t
          requests, self.requests = self.requests, []
        for want in requests:
          if want:
            if real.fault:
              self.driver.recover()
              real = self.real = self.driver.read()
            self.safety.disable()  # (clears a fault)
            why = self.safety.enable(target, age, real.q if real.finite() else None)
            log.info('enable: %s', why or 'enabled')
          else:
            self.safety.disable('disabled from the lobby' if self.clients else 'no lobby connected')
            log.info('disabled')
        step = self.safety.step(target, age, real.q, dt, real.fault)
        if step.q is not None:
          self.driver.command(step.q)
          if grip is not None and self.spec.gripper:
            self.driver.command_gripper(grip)
          holding = False
        elif not holding:
          self.driver.hold()
          holding = True
        if now - last_state >= 1.0 / self.state_rate:
          self.publish_state(real)
          last_state = now
        status = self.status_msg(step.stale)
        if status != self.status or now - last_status >= STATUS_S:
          self.server.publish(self.topics['status'], {'data': json.dumps(status)})
          self.status, last_status = status, now
        next_t += dt
        sleep = next_t - time.monotonic()
        if sleep > 0:
          self.stop.wait(sleep)
        else:
          next_t = time.monotonic()  # behind: no catching up
    finally:
      try:
        self.driver.hold()
      finally:
        self.driver.close()

  def publish_state(self, real):
    names, pos = list(self.spec.joints), list(real.q)
    vel = list(real.dq) if real.dq else []
    eff = list(real.tau) if real.tau else []
    g = self.spec.gripper
    if g and real.gripper is not None:
      for j in [g.joint, *g.mirror]:
        names.append(j)
        pos.append(real.gripper)
        if vel:
          vel.append(0.0)
        if eff:
          eff.append(0.0)
    t = time.time()
    self.server.publish(self.topics['real'], {
        'header': {'stamp': {'sec': int(t), 'nanosec': int((t % 1) * 1e9)}, 'frame_id': ''},
        'name': names, 'position': pos, 'velocity': vel, 'effort': eff})

  def status_msg(self, stale):
    s = self.safety
    return {'robot': self.spec.name, 'state': s.state, 'reason': s.reason, 'stale': stale,
            'error': round(s.error, 3) if s.state == ENABLED else None, 'clients': self.clients,
            'fault': self.real.fault if self.real else ''}
