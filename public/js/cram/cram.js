// The lobby's link to CRAM: a Jupyter kernel with CRAM in a Binder session of
// cramera's image (https://github.com/yxzhan/cramera), driven from this page
// over the session's REST API and the kernel's WebSocket. The Behavior
// panel's trees use it (lobby.Cram, lobby.CramRun in behavior/lobby.py).
//
// start(): the session from ?cram=<session URL with ?token=> if given, else
// the one this browser used last (localStorage) while it answers, else a new
// one launched on the hub (BinderHub's build API, as Go online does; ~1 min
// once the image is built). In it: a kernelspec `cram` (CRAM's venv with ROS
// sourced; written by a python3 kernel, the image has none), a cram kernel
// (one already running is reused: its world and imports stay), and
// lobby_cram.py run in it (CRAM's imports, pick_and_place; ~12 s the first
// time).
//
// run(code): a cell; resolves with {ok, value (the cell's application/json
// value, else its text/plain), stdout, error}.

import { launchBinder } from '../net/serverpanel.js';

const BINDER = { hub: 'https://binder.intel4coro.de', repo: 'gh/yxzhan/cramera/main', path: '' };
const KEY = 'mujoco-filament/cram-session';
const MODULE = new URL('./lobby_cram.py', import.meta.url);
const KERNEL = 'cram';
const KERNELSPEC = `
import json, os, pathlib
d = pathlib.Path.home() / ".local/share/jupyter/kernels/${KERNEL}"
d.mkdir(parents=True, exist_ok=True)
(d / "start.sh").write_text("""#!/bin/bash
source /opt/ros/\${ROS_DISTRO:-jazzy}/setup.bash
source /opt/cramera/ros-overlay/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
exec /opt/cramera/venv/bin/python -m ipykernel_launcher "$@"
""")
os.chmod(d / "start.sh", 0o755)
(d / "kernel.json").write_text(json.dumps({"argv": [str(d / "start.sh"), "-f", "{connection_file}"],
                                           "display_name": "CRAM", "language": "python"}))
`;

const store = {
  get() { try { return JSON.parse(localStorage.getItem(KEY)); } catch { return null; } },
  set(v) { try { localStorage.setItem(KEY, JSON.stringify(v)); } catch { /* (no storage) */ } },
};

// {base, token} of a session URL with ?token=.
function parseSession(href) {
  const u = new URL(href), token = u.searchParams.get('token');
  u.search = '';
  u.hash = '';
  const user = u.pathname.match(/^.*?\/user\/[^/]+\//);  // (a JupyterHub session: /user/<name>/…)
  u.pathname = user ? user[0] : u.pathname.replace(/[^/]*$/, '');
  return token ? { base: u.href, token } : null;
}

class Kernel {
  constructor(session, id) {
    this.session = session;
    this.id = id;
    this.pending = new Map();  // msg_id -> {resolve, out: {stdout, value, error}}
  }

  open() {
    const { base, token } = this.session;
    const url = new URL(`api/kernels/${this.id}/channels`, base);
    url.protocol = url.protocol.replace('http', 'ws');
    url.searchParams.set('token', token);
    return new Promise((resolve, reject) => {
      const ws = this.ws = new WebSocket(url);
      ws.onopen = () => resolve(this);
      ws.onerror = () => reject(new Error('the kernel\'s WebSocket failed'));
      ws.onclose = () => {
        this.closed = true;
        for (const p of this.pending.values()) p.resolve({ ok: false, error: 'the kernel connection closed', ...p.out });
        this.pending.clear();
        this.onclose?.();
      };
      ws.onmessage = (e) => this.message(JSON.parse(e.data));
    });
  }

  message(m) {
    const p = this.pending.get(m.parent_header?.msg_id);
    if (!p) return;
    const c = m.content, o = p.out;
    if (m.msg_type === 'stream') o.stdout += c.text;
    else if (m.msg_type === 'execute_result' || m.msg_type === 'display_data') o.value = c.data['application/json'] ?? c.data['text/plain'];
    else if (m.msg_type === 'error') o.error = `${c.ename}: ${c.evalue}\n${(c.traceback ?? []).join('\n').replace(/\x1b\[[0-9;]*m/g, '').slice(-2000)}`;
    else if (m.msg_type === 'status' && c.execution_state === 'idle') {
      this.pending.delete(m.parent_header.msg_id);
      p.resolve({ ok: !o.error, ...o });
    }
  }

  run(code) {
    if (this.closed) return Promise.resolve({ ok: false, error: 'the kernel connection is closed' });
    const id = crypto.randomUUID();
    return new Promise((resolve) => {
      this.pending.set(id, { resolve, out: { stdout: '', value: null, error: null } });
      this.ws.send(JSON.stringify({
        header: { msg_id: id, msg_type: 'execute_request', username: 'lobby', session: this.session.client, version: '5.3' },
        parent_header: {}, metadata: {}, channel: 'shell',
        content: { code, silent: false, store_history: false, user_expressions: {}, allow_stdin: false },
      }));
    });
  }
}

class Cram {
  constructor() {
    this.phase = 'off';  // off, launching, connecting, loading, ready, failed
    this.message = '';
    this.error = null;
    this.session = null;  // {base, token, client}
    this.kernel = null;
    this.starting = null;
  }

  state() {
    return { phase: this.phase, message: this.message, error: this.error, url: this.session?.base ?? null };
  }

  status(phase, message = '') {
    this.phase = phase;
    this.message = String(message).trim().slice(-300);
  }

  // The session's REST API: `path` under api/.
  async api(path, method = 'GET', body) {
    const { base, token } = this.session;
    const res = await fetch(new URL(`api/${path}?token=${encodeURIComponent(token)}`, base), {
      method, ...(body && { body: JSON.stringify(body), headers: { 'Content-Type': 'application/json' } }),
    });
    if (!res.ok) throw new Error(`${method} api/${path}: ${res.status}`);
    return res.status === 204 ? null : res.json();
  }

  async alive() {
    try { await this.api('status'); return true; } catch { return false; }
  }

  // Ready to run cells (once; again after a failure or a lost kernel).
  start() {
    if (this.phase === 'ready' && this.kernel && !this.kernel.closed) return Promise.resolve();
    this.starting ??= this.connect().then(
      () => { this.status('ready', this.session.base); this.error = null; },
      (err) => { this.error = err.message; this.status('failed', err.message); throw err; },
    ).finally(() => { this.starting = null; });
    return this.starting;
  }

  async connect() {
    this.error = null;
    const given = new URLSearchParams(location.search).get('cram');
    const candidates = [given && parseSession(given), store.get()].filter(Boolean);
    this.session = null;
    for (const s of candidates) {
      this.session = { ...s, client: crypto.randomUUID() };
      this.status('connecting', s.base);
      if (await this.alive()) break;
      this.session = null;
    }
    if (!this.session) {
      this.status('launching', 'starting a CRAM session on Binder');
      const href = await launchBinder(BINDER, (phase, msg) => this.status('launching', `${phase}: ${msg}`));
      this.session = { ...parseSession(href), client: crypto.randomUUID() };
    }
    store.set({ base: this.session.base, token: this.session.token });

    this.status('connecting', 'the CRAM kernel');
    let k = (await this.api('kernels')).find((x) => x.name === KERNEL);
    if (!k) {
      const specs = await this.api('kernelspecs');
      if (!specs.kernelspecs?.[KERNEL]) {
        const py = await this.api('kernels', 'POST', { name: 'python3' });
        const kp = await new Kernel(this.session, py.id).open();
        const r = await kp.run(KERNELSPEC);
        kp.ws.close();
        await this.api(`kernels/${py.id}`, 'DELETE').catch(() => {});
        if (!r.ok) throw new Error(`writing the cram kernelspec: ${r.error}`);
      }
      k = await this.api('kernels', 'POST', { name: KERNEL });
    }
    this.kernel = await new Kernel(this.session, k.id).open();
    this.kernel.onclose = () => { if (this.phase === 'ready') this.status('off', 'the kernel connection closed'); };

    this.status('loading', 'CRAM (lobby_cram.py)');
    const src = await fetch(MODULE).then((r) => { if (!r.ok) throw new Error(`lobby_cram.py: ${r.status}`); return r.text(); });
    const r = await this.kernel.run(src);
    if (!r.ok) throw new Error(`lobby_cram.py: ${r.error}`);
  }

  async run(code) {
    try {
      await this.start();
    } catch (err) {
      return { ok: false, error: err.message, stdout: '', value: null };
    }
    return this.kernel.run(code);
  }
}

export const cram = new Cram();
