"""A small rosbridge server (rosbridge v2's JSON over a WebSocket: advertise,
unadvertise, publish, subscribe, unsubscribe), enough for the lobby's ROS
bridge (public/js/ros/bridge.js) to talk to this process without ROS
installed: the lobby connects to it as to rosbridge_server.

Topics are only names here: what a client publishes goes to this process's
handlers (on(topic, fn)) and to the other clients subscribed to it, what
this process publishes (publish(), from any thread) to the subscribed
clients. No types are checked, no services offered (a call_service gets a
failed response), no ROS graph behind it: a real rosbridge in its place
works the same for the lobby.
"""

from __future__ import annotations

import asyncio
import json
import logging

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

log = logging.getLogger('rosbridge')


class RosbridgeServer:
  def __init__(self, host='0.0.0.0', port=9090):
    self.host, self.port = host, port
    self.loop = None
    self.clients = {}    # connection -> set of subscribed topics
    self.handlers = {}   # topic -> [fn(msg)]
    self.on_clients = None  # fn(count): a client came or went

  def on(self, topic, fn):
    self.handlers.setdefault(topic, []).append(fn)

  def publish(self, topic, msg):
    """From any thread: msg to every client subscribed to topic."""
    if self.loop:
      self.loop.call_soon_threadsafe(self.send, topic, msg, None)

  def send(self, topic, msg, sender):
    data = None
    for conn, topics in self.clients.items():
      if conn is sender or topic not in topics:
        continue
      data = data or json.dumps({'op': 'publish', 'topic': topic, 'msg': msg})
      asyncio.ensure_future(self.write(conn, data))

  async def write(self, conn, data):
    try:
      await conn.send(data)
    except ConnectionClosed:
      pass

  async def client(self, conn):
    self.clients[conn] = set()
    log.info('client %s connected', conn.remote_address)
    self.on_clients and self.on_clients(len(self.clients))
    try:
      async for text in conn:
        try:
          m = json.loads(text)
        except ValueError:
          continue
        op, topic = m.get('op'), m.get('topic')
        if op == 'subscribe' and topic:
          self.clients[conn].add(topic)
        elif op == 'unsubscribe' and topic:
          self.clients[conn].discard(topic)
        elif op == 'publish' and topic:
          msg = m.get('msg') or {}
          for fn in self.handlers.get(topic, ()):
            try:
              fn(msg)
            except Exception:
              log.exception('handler of %s', topic)
          self.send(topic, msg, conn)
        elif op == 'call_service':
          await self.write(conn, json.dumps({'op': 'service_response', 'id': m.get('id'), 'service': m.get('service'),
                                             'result': False, 'values': 'no services here'}))
        # advertise / unadvertise: nothing to keep
    except ConnectionClosed:
      pass
    finally:
      del self.clients[conn]
      log.info('client %s left', conn.remote_address)
      self.on_clients and self.on_clients(len(self.clients))

  async def run(self, stop=None):
    """Serves until `stop` (an asyncio.Event) is set, or forever."""
    self.loop = asyncio.get_running_loop()
    async with serve(self.client, self.host, self.port, max_size=2 ** 24, compression=None):
      log.info('rosbridge on ws://%s:%d', self.host, self.port)
      await (stop.wait() if stop else asyncio.Future())
