// One WebSocket to one rosbridge (rosbridge_suite's JSON protocol, v2:
// advertise / publish / subscribe), shared by everything that talks ROS
// through it: the robots on that address and the world's /clock
// (robot.js). Made when the first user needs it (RosWorld.bridge), closed
// with the last (release). It keeps the connection (retried every RETRY_MS
// while it has users) and the bookkeeping: a topic is advertised while one
// user publishes on it and subscribed while one listens, again after each
// reconnect.

const RETRY_MS = 2000;

// 'localhost:9090' -> 'ws://localhost:9090'; a ws:// or wss:// URL as it is.
export function bridgeUrl(text) {
  const t = text.trim();
  return /^wss?:\/\//.test(t) ? t : `ws://${t}`;
}

export class RosBridge {
  constructor(url, onEmpty) {
    this.url = url;               // as typed (host:port, or a ws:// URL)
    this.onEmpty = onEmpty;       // the last user left: the world forgets this bridge
    this.users = new Set();
    this.ws = null;
    this.status = 'off';          // off | connecting | connected | error
    this.error = '';
    this.sent = 0;
    this.received = 0;
    this.published = new Map();   // topic -> {type, users: Set}
    this.subscribed = new Map();  // topic -> {type, handlers: Set}
    this.onChange = new Set();    // () => …: the status changed
  }

  get connected() { return this.status === 'connected'; }

  // Bytes waiting to go out (camera frames are dropped above a limit).
  get buffered() { return this.ws?.bufferedAmount ?? 0; }

  acquire(user) {
    this.users.add(user);
    if (this.status === 'off') this.connect();
  }

  release(user) {
    this.users.delete(user);
    for (const [topic, p] of this.published) if (p.users.delete(user) && !p.users.size) this.unadvertise(topic);
    if (this.users.size) return;
    this.close('off');
    this.onEmpty?.(this);
  }

  changed() { for (const f of this.onChange) f(); }

  connect() {
    this.close('connecting');
    let ws;
    try {
      ws = this.ws = new WebSocket(bridgeUrl(this.url));
    } catch (err) {
      this.fail(err.message ?? String(err));
      return;
    }
    ws.onopen = () => {
      if (ws !== this.ws) return;
      this.status = 'connected';
      this.error = '';
      for (const [topic, { type }] of this.published) this.raw({ op: 'advertise', topic, type });
      for (const [topic, { type }] of this.subscribed) this.raw({ op: 'subscribe', topic, type, queue_length: 1 });
      this.changed();
    };
    ws.onmessage = (e) => { if (ws === this.ws) this.receive(e.data); };
    ws.onerror = () => { if (ws === this.ws) this.error = `no rosbridge at ${bridgeUrl(this.url)}`; };
    ws.onclose = () => { if (ws === this.ws) this.fail(this.error || 'connection closed'); };
    this.changed();
  }

  fail(message) {
    this.close('error');
    this.error = message;
    if (this.users.size) this.retry = setTimeout(() => this.users.size && this.connect(), RETRY_MS);
    this.changed();
  }

  close(status) {
    clearTimeout(this.retry);
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
      try { ws.close(); } catch { /* closing */ }
    }
    this.status = status;
    if (status === 'off') this.changed();
  }

  raw(msg) {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  // ---- topics ----

  // `user` publishes `type` on `topic` (advertised while one does).
  advertise(topic, type, user) {
    let p = this.published.get(topic);
    if (p && p.type !== type) { this.unadvertise(topic); p = null; }  // (a camera switched between JPEG and raw)
    if (!p) {
      this.published.set(topic, p = { type, users: new Set() });
      this.raw({ op: 'advertise', topic, type });
    }
    p.users.add(user);
  }

  // `user` stops publishing on `topic` (none: everyone).
  unadvertise(topic, user) {
    const p = this.published.get(topic);
    if (!p) return;
    if (user) p.users.delete(user);
    if (user && p.users.size) return;
    this.published.delete(topic);
    this.raw({ op: 'unadvertise', topic });
  }

  publish(topic, msg) {
    if (this.raw({ op: 'publish', topic, msg })) this.sent++;
  }

  // handler(msg) for every message on `topic`; returns the unsubscribe.
  subscribe(topic, type, handler) {
    let s = this.subscribed.get(topic);
    if (!s) {
      this.subscribed.set(topic, s = { type, handlers: new Set() });
      this.raw({ op: 'subscribe', topic, type, queue_length: 1 });
    }
    s.handlers.add(handler);
    return () => {
      s.handlers.delete(handler);
      if (s.handlers.size || this.subscribed.get(topic) !== s) return;
      this.subscribed.delete(topic);
      this.raw({ op: 'unsubscribe', topic });
    };
  }

  receive(text) {
    let msg;
    try { msg = JSON.parse(text); } catch { return; }
    if (msg.op !== 'publish') return;
    this.received++;
    for (const h of this.subscribed.get(msg.topic)?.handlers ?? []) h(msg.msg ?? {});
  }
}
