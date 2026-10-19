// Where the page's server and data are. The page's code (this directory)
// may come from somewhere else than its data: GitHub Pages serves the code
// and a small part of the model library, a server (server/server.py, e.g. on
// Binder) the whole library and the shared world.
//
// The server, the first of:
//   ?local          none: local mode on the page's own files
//   ?server=<url>   that one; remembered in this browser. A Binder URL
//                   carries its token (…/mujoco/?token=…), which then goes
//                   on every request, WebSockets included: the page is on
//                   another site, its login cookie would not be sent.
//   remembered      the last one connected to, while it still answers
//   the page's own  when it answers /server.json (server.py serves the page)
// With none the page runs MuJoCo itself on the files next to it.
//
// With a server every data file (assets.json, thumbnails, mjcf_assets/,
// pretrained_policies/) comes from it: dataUrl.

const STORE = 'mujoco-filament/server';

const params = new URLSearchParams(location.search);

const remember = (url) => {
  try {
    if (url) localStorage.setItem(STORE, url);
    else localStorage.removeItem(STORE);
  } catch { /* no storage */ }
};
const remembered = () => {
  try { return localStorage.getItem(STORE); } catch { return null; }
};

// '…/mujoco/?token=abc' -> {base: '…/mujoco/', token: 'abc'}
function parse(url) {
  const u = new URL(url, location.href);
  const token = u.searchParams.get('token');
  u.search = '';
  u.hash = '';
  if (!u.pathname.endsWith('/')) u.pathname += '/';
  return { base: u.href, token };
}

const withToken = (url, token) => {
  if (token) url.searchParams.set('token', token);
  return url;
};

async function probe({ base, token }) {
  try {
    const res = await fetch(withToken(new URL('server.json', base), token), { cache: 'no-store' });
    return res.ok && (res.headers.get('Content-Type') ?? '').includes('json') ? await res.json() : null;
  } catch {
    return null;
  }
}

// {base, token, info (its server.json), url (base with token), remote
// (not the page's own origin)}, or null for local mode. `problem`: why a
// server asked for is not used.
export let server = null;
export let problem = null;

if (!params.has('local')) {
  const asked = params.get('server'), saved = remembered();
  const own = { base: new URL('.', location.href).href, token: null };
  for (const [cand, from] of [[asked, 'asked'], [!asked && saved, 'saved'], [own, 'own']]) {
    if (!cand) continue;
    const s = typeof cand === 'string' ? parse(cand) : cand;
    const info = await probe(s);
    if (info) {
      const url = withToken(new URL(s.base), s.token).href;
      server = { ...s, info, url, remote: new URL(s.base).origin !== location.origin };
      if (from !== 'own') remember(url);
      break;
    }
    if (from === 'asked') { problem = `no server answers at ${s.base}`; break; }
    if (from === 'saved') remember(null);  // gone (a Binder session ends)
  }
}

// A data file (`rel` relative to public/, as in assets.json) as a URL.
export function dataUrl(rel) {
  return server ? withToken(new URL(rel, server.base), server.token).href : rel;
}

// A request to the server: `path` with the `query` parameters.
export function serverUrl(path, query = {}) {
  const u = withToken(new URL(path, server.base), server.token);
  for (const [k, v] of Object.entries(query)) u.searchParams.set(k, v);
  return u.href;
}

// The server's WebSocket.
export function wsUrl(path = 'ws') {
  const u = withToken(new URL(path, server.base), server.token);
  u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
  return u.href;
}

// This page with no server parameters (and the rest kept).
function pageUrl(set = {}) {
  const u = new URL(location.href);
  u.searchParams.delete('server');
  u.searchParams.delete('local');
  for (const [k, v] of Object.entries(set)) u.searchParams.set(k, v);
  return u.href;
}

// Connects to the server at `url` (reloading the page), or goes local with
// null.
export function useServer(url) {
  if (url) parse(url);  // throws on a malformed one
  else remember(null);
  location.href = url ? pageUrl({ server: url }) : pageUrl({ local: '1' });
}

// A link to this page on the current server, for someone else to join.
export const shareUrl = () => (server?.remote ? pageUrl({ server: server.url }) : location.href);
