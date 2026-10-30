// Who drives which robot in a shared world (server/control.py,
// docs/control.md). This page is a client of the server: an id kept by its
// tab (so a reload is the same client and finds its robots still held) and a
// name kept by the browser. A robot with controllers is free or held by one
// client; the first robot op from a page takes a free one, and only its
// holder may drive it. Someone else may take it over once the holder has
// been idle on it for a while (take_after, 5 s; the hold lets go by itself
// then too).
//
// The server sends who is connected (clients) and the holds with their idle
// times (on changes and every second); `onChange` (main.js) re-renders the
// panels that show them.

import { newId } from '../world/stage.js';  // (crypto.randomUUID is not there over http by IP)

const ID_KEY = 'robot-lobby/client-id';  // sessionStorage: this tab
const NAME_KEY = 'robot-lobby/name';     // localStorage: this browser

function stored(storage, key, make) {
  try {
    const v = storage.getItem(key);
    if (v) return v;
    const fresh = make();
    storage.setItem(key, fresh);
    return fresh;
  } catch {
    return make();  // storage blocked
  }
}

// A client's mark: an icon of its kind of device (PC, Mobile, VR, Script)
// in its color, as a <svg class="who">.
const ICONS = {
  PC: '<rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/>',
  Mobile: '<rect x="6.5" y="2" width="11" height="20" rx="2.5"/><path d="M11 18h2"/>',
  VR: '<path d="M4 7h16a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1h-5l-3-3-3 3H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z"/><circle cx="8" cy="11.5" r="1.5"/><circle cx="16" cy="11.5" r="1.5"/>',
  Script: '<path d="M4 17l6-5-6-5M12 19h8"/>',
};
export function mark(client) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', 'who');
  svg.innerHTML = ICONS[client?.kind] ?? ICONS.PC;
  svg.style.color = client?.color ?? '#888';
  if (client?.kind) svg.setAttribute('aria-label', client.kind);
  return svg;
}

export class Control {
  constructor(app) {
    this.app = app;
    this.id = stored(sessionStorage, ID_KEY, () => `c-${newId().slice(0, 13)}`);
    this.name = (() => { try { return localStorage.getItem(NAME_KEY) ?? ''; } catch { return ''; } })();
    this.you = null;            // this page's id as the server knows it
    this.clients = new Map();   // id -> {id, name, color, here, device ('iPhone · Safari', from its User-Agent), kind (PC | Mobile | VR | Script)}
    this.holds = new Map();     // robot netId -> {client, idle (s, when it came), at (performance.now())}
    this.takeAfter = 15;
    this.onChange = null;
  }

  // What the page sends first on every connection (net.js).
  hello() {
    return { id: this.id, name: this.name, touch: navigator.maxTouchPoints > 1 };  // touch: iPadOS says it is a Mac
  }

  rename(name) {
    this.name = name.trim().slice(0, 40);
    try { localStorage.setItem(NAME_KEY, this.name); } catch { /* no storage */ }
    this.app.net?.send('rename', { name: this.name });
  }

  setClients(msg) {
    this.you = msg.you;
    this.clients = new Map(msg.clients.map((c) => [c.id, c]));
    this.onChange?.('clients');
  }

  setHolds(msg) {
    const now = performance.now();
    this.takeAfter = msg.take_after ?? this.takeAfter;
    const before = [...this.holds].map(([id, h]) => `${id}:${h.client}:${this.idle(id) >= this.takeAfter}`).join();
    this.holds = new Map(Object.entries(msg.holds).map(([id, h]) => [+id, { ...h, at: now }]));
    const after = [...this.holds].map(([id, h]) => `${id}:${h.client}:${this.idle(id) >= this.takeAfter}`).join();
    this.onChange?.(before === after ? 'idle' : 'holds');
  }

  // The client holding robot r (null: free), and whether that is this page.
  holder(r) {
    const h = this.holds.get(r?.netId);
    return h ? this.clients.get(h.client) ?? { id: h.client, name: 'someone', color: '#888' } : null;
  }

  mine(r) {
    return this.holds.get(r?.netId)?.client === this.you;
  }

  // Held by someone else: this page must not drive it.
  blocked(r) {
    const h = this.holds.get(r?.netId);
    return !!h && h.client !== this.you;
  }

  // s since its holder last drove it.
  idle(netId) {
    const h = this.holds.get(netId);
    return h ? h.idle + (performance.now() - h.at) / 1000 : 0;
  }

  canTake(r) {
    const h = this.holder(r);
    return !!h && !this.mine(r) && (!h.here || this.idle(r.netId) >= this.takeAfter);
  }

  take(r) { this.app.net?.send('take', { id: r.netId }); }
  release(r) { this.app.net?.send('hold', { id: r.netId, on: false }); }
}
