// The Server panel: where the world runs (endpoint.js). Local mode runs
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

import { problem, server, shareUrl, useServer } from './endpoint.js';

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
    this.body = root.querySelector('.body');
    this.models = models;
    this.deploy = {};
    this.launching = false;
    this.log = null;      // the last Binder launch: [phase, message] lines
    this.render();
    if (!server) loadDeploy().then((d) => { this.deploy = d; this.render(); });
  }

  render() {
    const b = this.body;
    b.replaceChildren();
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

    const bar = el('div', { className: 'bar' });
    if (server?.remote) {
      const copy = el('button', { textContent: 'Copy link', title: 'a link to this page on this server, for someone else to join (it holds the session\'s token)' });
      copy.onclick = async () => {
        try {
          await navigator.clipboard.writeText(shareUrl());
          copy.textContent = 'Copied';
        } catch {
          prompt('Link to join this world:', shareUrl());
        }
      };
      bar.append(copy);
    }
    if (server) {
      const local = el('button', { textContent: 'Go local', title: 'leave the server; physics runs in this browser' });
      local.onclick = () => useServer(null);
      bar.append(local);
    }
    if (bar.childElementCount) b.append(bar);

    const binder = this.deploy.binder;
    if (binder?.hub && binder?.repo && !server) {
      b.append(el('div', { className: 'sub', textContent: 'Binder' }));
      const start = el('button', { className: 'primary', textContent: 'Start a server on Binder',
        title: `${binder.hub}: ${binder.repo}`, disabled: this.launching });
      start.onclick = () => this.launch(binder);
      b.append(el('div', { className: 'hint', textContent:
        'A new session of its own: building it the first time takes minutes, starting it again tens of seconds. ' +
        'It ends after a while without anyone connected.' }), el('div', { className: 'bar' }, start));
      if (this.log) b.append(el('div', { className: 'log' }, ...this.log.map(([phase, msg]) =>
        el('div', { className: phase === 'failed' ? 'error' : '', textContent: `${phase}${msg ? `: ${msg.trim()}` : ''}` }))));
    }

    b.append(el('div', { className: 'sub', textContent: 'Join a server' }));
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
