// Embedded in another page (cramera's Shared Sim tab), which opens this page in
// a frame with ?embed=<its own address>: the two keep their address
// parameters the same, through messages.
//
//   lobby -> page  {type: 'robot-lobby:url', params}     this page's parameters,
//                  as it opens and whenever it changes them (publishUrl)
//   page -> lobby  {type: 'robot-lobby:params', params}  open with these instead
//                  (the page's back and forward)
//
// `params` is [[name, value], ...] without `embed` and `online` (the
// embedding page's own: where it is, and to start a shared world). Messages
// go only to the origin ?embed= names and are taken only from it, from the
// parent frame: a `server` parameter carries its access token.
// The links this page gives others (Share panel, a stage's link) open the
// embedding page with the same parameters (outward), not this page alone.

const OWN = ['embed', 'online'];

const params = new URLSearchParams(location.search);

const embedUrl = (() => {
  try {
    const u = params.get('embed') && new URL(params.get('embed'));
    return u && /^https?:$/.test(u.protocol) ? u : null;
  } catch {
    return null;
  }
})();

// Whether this page is in the frame of the page ?embed= names.
export const embedded = !!embedUrl && window.parent !== window;

// This page's parameters that the two share.
const shared = (search) => [...new URLSearchParams(search)].filter(([k]) => !OWN.includes(k));

// Tells the embedding page this page's parameters (after a history.replaceState).
export function publishUrl() {
  if (!embedded) return;
  window.parent.postMessage({ type: 'robot-lobby:url', params: shared(location.search) }, embedUrl.origin);
}

// `url` (a link to this page) as a link to the embedding page with the same
// parameters, when embedded; else `url` itself.
export function outward(url) {
  if (!embedded) return url;
  const from = new URL(url, location.href);
  const to = new URL(embedUrl.href);
  for (const [k, v] of shared(from.search)) to.searchParams.set(k, v);
  to.hash = from.hash;
  return to.href;
}

if (embedded) {
  window.addEventListener('message', (e) => {
    if (e.origin !== embedUrl.origin || e.source !== window.parent) return;
    if (e.data?.type !== 'robot-lobby:params' || !Array.isArray(e.data.params)) return;
    const next = new URL(location.href);
    next.search = '';
    for (const [k, v] of e.data.params) if (!OWN.includes(k)) next.searchParams.append(String(k), String(v));
    // As the embedding page opens the frame: ?online without a server or ?local.
    if (!next.searchParams.has('server') && !next.searchParams.has('local')) next.searchParams.set('online', '');
    next.searchParams.set('embed', embedUrl.href);
    if (next.search !== location.search) location.replace(next.href);
    else publishUrl();  // has them already: says so
  });
  publishUrl();
}
