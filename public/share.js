// The Share panel: this page on another device (a phone, a headset) without
// typing its long URL. The pairing service (join.aicor.dev; `?pair=<url>`
// overrides it) holds code -> link for a while: the other device opens the
// service and types the short code, or scans the QR code of
// <service>/<code>, and is sent on to the link. A code is asked for when the
// panel is shown; New code asks again. Ported from bambot's lib/shareLink.ts.
//
// The link: on a server of its own (endpoint.js: ?server=…), this page with
// that server, token included, so the other device joins the same world;
// on the page's own server, this page with the token it was opened with
// (behind Jupyter's proxy the token can drop out of the address bar once the
// proxy has set its cookie, so it is remembered for the tab); in local mode
// just this page, where the other device gets a world of its own.

import { server, shareUrl } from './endpoint.js';

const TOKEN_PARAM = 'token';
const TOKEN_KEY = 'robot-lobby/session-token';  // sessionStorage
export const PAIRING_SERVICE = 'https://join.aicor.dev';

// The access token this page was opened with, remembered for the tab.
function sessionToken() {
  const token = new URLSearchParams(location.search).get(TOKEN_PARAM);
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    return token ?? sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return token;  // storage blocked (private mode)
  }
}

// The link another device should open.
function shareTarget(token) {
  if (server?.remote) return shareUrl();
  const url = new URL(location.href);
  url.hash = '';
  if (token && server) url.searchParams.set(TOKEN_PARAM, token);
  return url.href;
}

// Why the link may not work on another device, or null.
function linkWarning(token) {
  const url = new URL(location.href);
  if (['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) {
    return 'This page is on localhost, which another device cannot reach. Open it by this machine\'s network address (over HTTPS) before sharing.';
  }
  if (server && !server.remote && !token && /\/(user|proxy)\//.test(url.pathname)) {
    return 'This page was opened without an access token, so the other device will be asked to sign in. Open it with ?token=… to share it.';
  }
  return null;
}

const pairingService = () => (new URLSearchParams(location.search).get('pair') || PAIRING_SERVICE).replace(/\/+$/, '');
const typedAddress = (service) => service.replace(/^https?:\/\//, '');
const codeLink = (service, code) => `${service}/${encodeURIComponent(code)}`;

// {code, expiresAt (ms since the epoch, or null)} for a code leading to `target`.
async function requestPairingCode(service, target) {
  const res = await fetch(`${service}/api/pairings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: target }),
    credentials: 'omit',
  });
  if (!res.ok) throw new Error(`the pairing service answered ${res.status}`);
  const answer = await res.json();
  if (!answer?.code) throw new Error('the pairing service sent no code');
  return { code: String(answer.code), expiresAt: Date.parse(answer.expires_at) || null };
}

const remaining = (ms) => {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

export class SharePanel {
  constructor(root) {
    this.root = root;
    this.body = root.querySelector('.body');
    this.service = pairingService();
    this.token = sessionToken();
    this.state = { status: 'idle' };  // idle | loading | ready {pairing} | error {message}
    this.request = 0;                 // a code for a superseded request is dropped
    this.timer = null;
    this.render();
    // Ask for a code when the panel is shown and has none that is still good.
    new MutationObserver(() => {
      if (root.hidden) return;
      const p = this.state.pairing;
      if (this.state.status !== 'loading' && !(p && (p.expiresAt === null || p.expiresAt > Date.now()))) this.pair();
    }).observe(root, { attributes: true, attributeFilter: ['hidden'] });
    if (!root.hidden) this.pair();
  }

  async pair() {
    const request = ++this.request;
    this.state = { status: 'loading' };
    this.render();
    try {
      const pairing = await requestPairingCode(this.service, shareTarget(this.token));
      if (request !== this.request) return;
      this.state = { status: 'ready', pairing };
    } catch (err) {
      if (request !== this.request) return;
      this.state = { status: 'error', message: err.message ?? String(err) };
    }
    this.render();
  }

  render() {
    clearInterval(this.timer);
    const { service, state } = this;
    const target = shareTarget(this.token);
    const b = this.body;
    b.replaceChildren();
    b.append(el('div', { className: 'hint', textContent: server
      ? 'Whoever opens it joins this world: they see and move the same robots.'
      : 'Local mode: whoever opens it gets a world of their own.' }));

    const p = state.pairing;
    const code = el('div', { className: 'code', textContent: p ? p.code : state.status === 'loading' ? '····' : '––––' });
    const qr = el('div', { className: 'qr' });
    const note = el('div', { className: state.status === 'error' ? 'note error' : 'note' });
    const link = el('a', { href: service, target: '_blank', rel: 'noopener noreferrer', textContent: typedAddress(service) });
    b.append(el('div', { className: 'pair' },
      el('div', { className: 'hint' }, 'On the other device open ', link, ' and enter'), code, qr, note));
    if (p) {
      import('./vendor/qrcode/qrcode.mjs').then(({ default: qrcode }) => {
        const q = qrcode(0, 'M');
        q.addData(codeLink(service, p.code));
        q.make();
        qr.innerHTML = q.createSvgTag({ cellSize: 4, margin: 2, scalable: true });
      }).catch(() => qr.remove());
    }
    const tick = () => {
      const expired = p?.expiresAt != null && p.expiresAt <= Date.now();
      code.classList.toggle('expired', expired);
      qr.hidden = !p || expired;
      note.textContent = state.status === 'loading' ? `asking ${typedAddress(service)} for a code…`
        : state.status === 'error' ? `No code: ${state.message}`
          : !p || p.expiresAt == null ? ''
            : expired ? 'this code has expired' : `valid for ${remaining(p.expiresAt - Date.now())}`;
      if (expired) clearInterval(this.timer);
    };
    tick();
    if (p?.expiresAt != null) this.timer = setInterval(tick, 1000);

    const again = el('button', { textContent: '↻ New code', disabled: state.status === 'loading' });
    again.onclick = () => this.pair();
    b.append(el('div', { className: 'bar' }, again));

    const input = el('input', { type: 'text', readOnly: true, value: target });
    input.onfocus = () => input.select();
    const copy = el('button', { textContent: 'Copy' });
    copy.onclick = async () => {
      try {
        await navigator.clipboard.writeText(target);
        copy.textContent = 'Copied';
      } catch {
        input.focus();
        copy.textContent = 'Press Ctrl+C';
      }
      setTimeout(() => { copy.textContent = 'Copy'; }, 1800);
    };
    b.append(el('div', { className: 'bar' }, input, copy));
    const warning = linkWarning(this.token);
    if (warning) b.append(el('div', { className: 'warn', textContent: warning }));
  }
}
