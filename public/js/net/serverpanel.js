// The Share panel's lower part (share.js makes its upper one, the pairing
// code): who is here, and where the world runs (endpoint.js). Local mode runs
// MuJoCo in this browser on the models next to the page; a server
// (server/server.py) runs one world shared by every browser connected to it,
// with its whole model library.
//
// A server is joined by its URL, or started on Binder from here when the
// page knows a Binder repository for it (json/deploy.json, written by
// scripts/build_pages.mjs):
//   {"binder": {"hub": "https://mybinder.org", "repo": "gh/<owner>/<repo>/<ref>", "path": "robot-lobby/"}}
// BinderHub's build API reports the build and launch as server-sent events
// and ends with the session's URL and token; the world server runs behind
// the session's jupyter-server-proxy at `path` (binder/jupyter_server_config.py),
// started on the first request. The hub has to answer the page's origin
// (Access-Control-Allow-Origin, docs/deploy.md).
//
// Go online (the toolbar's button in local mode, and the panel's): the
// launch runs in the background, the page usable meanwhile, its phases in
// the status line. Once the session is ready the local scene is saved in
// this browser (My stages, app.carryScene in main.js) and the page reloads
// onto the server, which loads it there (takeCarried).
//
// On a server: who is connected (net/control.js), with the robots each
// drives, and this page's name.

import { problem, server, useServer } from '../app/endpoint.js';
import { mark } from './control.js';

// Start a server on Binder from the page.
const BINDER_LAUNCH = true;

// sessionStorage: the id of the stage (My stages) a Go online carries over,
// for the page once it is on the server.
const CARRY_KEY = 'mujoco-filament/carry';

// The stage id Go online left for this page, once (null: none).
export function takeCarried() {
  try {
    const id = sessionStorage.getItem(CARRY_KEY);
    sessionStorage.removeItem(CARRY_KEY);
    return id;
  } catch {
    return null;
  }
}

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

async function loadDeploy() {
  try {
    const res = await fetch('json/deploy.json', { cache: 'no-store' });
    return res.ok ? await res.json() : {};
  } catch {
    return {};
  }
}

// Starts a Binder session for `binder` ({hub, repo, path}); `progress(phase,
// message)` follows it. Resolves with the server's URL (token included).
export function launchBinder({ hub, repo, path = 'robot-lobby/' }, progress) {
  return new Promise((resolve, reject) => {
    const events = new EventSource(new URL(`build/${repo}`, hub.replace(/\/?$/, '/')));
    events.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      progress(msg.phase, msg.message ?? '');
      if (msg.phase === 'ready') {
        events.close();
        const url = new URL(path, msg.url);
        url.searchParams.set('token', msg.token);
        resolve(url.href);
      } else if (msg.phase === 'failed') {
        events.close();
        reject(new Error(msg.message || 'the build failed'));
      }
    };
    events.onerror = () => {
      events.close();
      reject(new Error('no answer from AICOR VRB'));
    };
  });
}

export class ServerPanel {
  // `online`: the toolbar's Go online button, shown in local mode when the
  // page knows a Binder repository.
  constructor(app, root, { models = 0, online = null } = {}) {
    this.app = app;
    this.body = root.querySelector('.server');
    this.models = models;
    this.online = online;
    this.deploy = {};
    this.launching = false;
    this.log = null;      // the last Binder launch: [phase, message] lines
    this.render();
    if (online) online.onclick = () => this.launch(this.deploy.binder);
    if (!server && BINDER_LAUNCH) loadDeploy().then((d) => { this.deploy = d; this.render(); });
  }

  get binder() {
    const b = this.deploy.binder;
    return BINDER_LAUNCH && b?.hub && b?.repo && !server ? b : null;
  }

  render() {
    const b = this.body;
    b.replaceChildren();
    if (server && this.app.control) this.renderPeople(b);
    b.append(el('div', { className: 'sub', textContent: 'Server' }));
    if (server) {
      const where = server.remote ? new URL(server.base).host : 'this page\'s server';
      b.append(el('div', { className: 'hint', textContent:
        `Shared world on ${where} (MuJoCo ${server.info.mujoco}): every browser connected to it sees and moves the same robots.` }));
    } else {
      b.append(el('div', { className: 'hint', textContent:
        `Local: physics runs in this browser, on the ${this.models} models that come with this page.` }));
    }
    if (problem) b.append(el('div', { className: 'note', textContent: `${problem}: running locally.` }));
    if (server) {
      const local = el('button', { textContent: 'Go local', title: 'leave the server; physics runs in this browser' });
      local.onclick = () => useServer(null);
      b.append(el('div', { className: 'bar' }, local));
    }

    const binder = this.binder;
    if (this.online) {
      this.online.hidden = !binder;
      this.online.disabled = this.launching;
      this.online.querySelector('.label').textContent = this.launching ? 'Starting the shared lobby…' : 'Start a shared lobby on AICOR VRB';
    }
    if (binder) {
      b.append(el('div', { className: 'sub minor', textContent: 'AICOR VRB' }));
      const start = el('button', { className: 'primary', textContent: this.launching ? 'Starting the shared lobby…' : 'Start a shared lobby on AICOR VRB',
        disabled: this.launching });
      start.onclick = () => this.launch(binder);
      b.append(el('div', { className: 'hint', textContent:
        'Takes this scene online with the full asset library, for others to join. Starting takes seconds to a few minutes.' }),
      el('div', { className: 'bar' }, start));
      if (this.log) b.append(el('div', { className: 'log' }, ...this.log.map(([phase, msg]) =>
        el('div', { className: phase === 'failed' ? 'error' : '', textContent: `${phase}${msg ? `: ${msg.trim()}` : ''}` }))));
    }

    b.append(el('div', { className: 'sub minor', textContent: 'Join a server' }));
    const input = el('input', { type: 'url', placeholder: 'https://…/robot-lobby/?token=…', value: server?.remote ? server.url : '' });
    const go = el('button', { textContent: 'Connect' });
    const connect = () => {
      try {
        if (input.value.trim()) useServer(input.value.trim());
      } catch {
        input.setCustomValidity('not a URL');
        input.reportValidity();
      }
    };
    go.onclick = connect;
    input.onkeydown = (e) => { input.setCustomValidity(''); if (e.key === 'Enter') connect(); };
    b.append(el('div', { className: 'bar' }, input, go),
      el('div', { className: 'hint', textContent: 'server/server.py\'s address; on AICOR VRB its proxy path with the session\'s token.' }));
  }

  // Again for new clients or holds, but not under the typing of a name.
  refresh() {
    const f = document.activeElement;
    if (f?.tagName === 'INPUT' && this.body.contains(f)) { this.stale = true; return; }
    this.render();
  }

  renderPeople(b) {
    const control = this.app.control;
    b.append(el('div', { className: 'sub', textContent: 'People' }));
    const me = control.clients.get(control.you);
    const name = el('input', { type: 'text', maxLength: 40, placeholder: me?.name ?? 'Your name', value: control.name,
      title: 'How others see you: next to the robots you drive' });
    name.onchange = () => control.rename(name.value);
    name.onkeydown = (e) => { if (e.key === 'Enter') name.blur(); };
    name.onblur = () => { if (this.stale) { this.stale = false; this.render(); } };
    b.append(el('label', { className: 'field' }, el('span', { textContent: 'Your name' }), name));
    const tags = el('input', { type: 'checkbox', checked: !!this.app.presence?.showTags });
    tags.onchange = () => this.app.presence?.setTags(tags.checked);
    b.append(el('label', { className: 'field', title: 'A tag over every robot someone drives: their device and name ("you" for yours)' },
      el('span', { textContent: 'Tags' }), el('span', {}, tags, ' names over the robots being driven')));
    const robots = this.app.teleop?.controllable() ?? [];
    const list = el('div', { className: 'people' });
    for (const c of control.clients.values()) {
      const drives = robots.filter((r) => control.holder(r)?.id === c.id).map((r) => `${r.prefix.slice(0, -1)} ${r.label}`);
      list.append(el('div', { className: 'person' + (c.here ? '' : ' gone') },
        mark(c),
        el('span', { className: 'name', textContent: c.name + (c.id === control.you ? ' (you)' : '') + (c.here ? '' : ' (gone)') }),
        el('span', { className: 'device', textContent: c.device ?? '' }),
        el('span', { className: 'note', textContent: drives.length ? `drives ${drives.join(', ')}` : '' })));
    }
    b.append(list, el('div', { className: 'hint', textContent:
      'A robot is driven by one person at a time: driving a free one makes it yours (Teleop panel); ' +
      `someone else can take it over once you have left it alone for ${control.takeAfter} s.` }));
  }

  async launch(binder) {
    if (!binder || this.launching) return;
    this.launching = true;
    this.log = [];
    const status = document.getElementById('status');
    const progress = (phase, message) => {
      // Build logs come line by line: keep the phases and the last line.
      if (this.log.at(-1)?.[0] === phase) this.log[this.log.length - 1] = [phase, message];
      else this.log.push([phase, message]);
      status.textContent = `starting the shared lobby: ${phase}${message ? ` · ${message.trim()}` : ''}`;
      this.render();
    };
    progress('starting', 'on AICOR VRB');
    try {
      const url = await launchBinder(binder, progress);
      // The scene as it is now (it went on while the session started).
      progress('ready', 'taking this scene along…');
      const id = await this.app.carryScene?.();
      try {
        if (id) sessionStorage.setItem(CARRY_KEY, id);
      } catch { /* no storage: the server starts empty, the stage is in My stages */ }
      useServer(url);
    } catch (err) {
      this.launching = false;
      progress('failed', err.message ?? String(err));
    }
  }
}
