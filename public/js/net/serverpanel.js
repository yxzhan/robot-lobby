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
// started on the first request.
//
// On a server: who is connected (net/control.js), with the robots each
// drives, and this page's name.

import { problem, server, useServer } from '../app/endpoint.js';
import { mark } from './control.js';

// Start a server on Binder from the page: off until BinderHub's build API
// answers other origins (TODOS.md, docs/deploy.md).
const BINDER_LAUNCH = false;

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
      reject(new Error(`no answer from ${hub}`));
    };
  });
}

export class ServerPanel {
  constructor(app, root, { models = 0 } = {}) {
    this.app = app;
    this.body = root.querySelector('.server');
    this.models = models;
    this.deploy = {};
    this.launching = false;
    this.log = null;      // the last Binder launch: [phase, message] lines
    this.render();
    if (!server && BINDER_LAUNCH) loadDeploy().then((d) => { this.deploy = d; this.render(); });
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
        `Local: physics runs in this browser, on the ${this.models} models that come with this page. ` +
        'A server has the whole library and a world shared with everyone connected.' }));
    }
    if (problem) b.append(el('div', { className: 'note', textContent: `${problem}: running locally.` }));
    if (server) {
      const local = el('button', { textContent: 'Go local', title: 'leave the server; physics runs in this browser' });
      local.onclick = () => useServer(null);
      b.append(el('div', { className: 'bar' }, local));
    }

    const binder = this.deploy.binder;
    if (BINDER_LAUNCH && binder?.hub && binder?.repo && !server) {
      b.append(el('div', { className: 'sub minor', textContent: 'Binder' }));
      const start = el('button', { className: 'primary', textContent: 'Start a server on Binder',
        title: `${binder.hub}: ${binder.repo}`, disabled: this.launching });
      start.onclick = () => this.launch(binder);
      b.append(el('div', { className: 'hint', textContent:
        'A new session of its own: building it the first time takes minutes, starting it again tens of seconds. ' +
        'It ends after a while without anyone connected.' }), el('div', { className: 'bar' }, start));
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
      el('div', { className: 'hint', textContent: 'server/server.py\'s address; on Binder its proxy path with the session\'s token.' }));
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
    this.launching = true;
    this.log = [];
    const progress = (phase, message) => {
      // Build logs come line by line: keep the phases and the last line.
      if (this.log.at(-1)?.[0] === phase) this.log[this.log.length - 1] = [phase, message];
      else this.log.push([phase, message]);
      this.render();
    };
    try {
      useServer(await launchBinder(binder, progress));
    } catch (err) {
      this.launching = false;
      progress('failed', err.message ?? String(err));
    }
  }
}
