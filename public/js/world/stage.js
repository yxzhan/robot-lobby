// Stages: what is spawned and how it stands, saved, published, shared and
// exported the same way in local and network mode.
//
// A stage is a folder named by a random id (crypto.randomUUID), so two
// stages of the same name never collide; its name is in stage.json:
//   stage.json     {version, id, name, saved, items, physics, view; published
//                  ones also author, published}
//   thumbnail.jpg  the main view when it was saved, 256x256
//   ik.json        IK setups in use, by robot file (keyed like json/ik_defaults.json)
//   base.json      mobile base setups in use, likewise (json/base_defaults.json)
//   cameras.json   camera setups in use, by cameraKey (json/camera_defaults.json)
// An item: {rel, label, pos, yaw, use (controllers switched on, robots only),
// cameras (the robot's switched-on cameras), state: {qpos: {joint: value, or
// the list of a free or ball joint}, ctrl: {actuator: value}, mocap: {body:
// {pos, quat}}}}, names without the robot's prefix ('#k' for the k-th joint,
// actuator or body of the robot when it has no name). `view` is the orbit
// camera ({center, dist, azim, elev}). Automatic controller and camera
// setups are left out: they are found again from the model.
// Save and Publish ask which parts to keep (STAGE_PARTS: joint state,
// controllers on, IK and base setups, cameras, physics, view); the others
// are left out of the bundle (withParts), and a stage loads without them.
//
// Where stages are kept, as the Stage panel lists them:
// - Mine: in this browser, in either mode (Save, Import): IndexedDB, one
//   record per stage holding the same files (BrowserStages). In network
//   mode the server makes the bundle of its world (/api/stages/capture).
// - Community, read-only: the published ones, in the library's
//   saved_stage/<id>/. On a server, its shared library (ServerStages):
//   Publish (network mode only) writes a stage there, with its author's
//   name if given (server.py publish_stage; not committed: the library's
//   git repository is kept by hand). In local mode, the stages of the
//   library next to the page, listed in public/json/stages.json by
//   scripts/vendor.mjs (LibraryStages).
// Download exports a stage as one JSON file, a bundle: {stage, files:
// {'ik.json', 'base.json', 'cameras.json'}, thumbnail (a data URL)}; Import
// takes such a file into Mine.

import { dataUrl, serverUrl } from '../app/endpoint.js';

export const STAGE_VERSION = 1;
export const STAGE_DIR = 'mjcf_assets/saved_stage';
const FILES = ['ik.json', 'base.json', 'cameras.json'];
const QPOS_SIZE = { 0: 7, 1: 4 };  // free, ball joints; slide and hinge have one
const round = (v) => +v.toFixed(6);

// A robot's joints, actuators and mocap bodies in the world, by local name.
// `at`: the robot's offsets in the state arrays (main.js stateLayout).
function robotTables(mujoco, model, robot, at) {
  const T = mujoco.mjtObj, { nq } = robot.visual.dims;
  const local = (obj, id, k) => {
    const n = mujoco.mj_id2name(model, obj, id) ?? '';
    return n.startsWith(robot.prefix) ? n.slice(robot.prefix.length) : n || `#${k}`;
  };
  const joints = [], ctrls = [], mocaps = [];
  let first = -1;
  for (let j = 0; j < model.njnt; j++) {
    const adr = model.jnt_qposadr[j];
    if (adr < at.qpos || adr >= at.qpos + nq) continue;
    if (first < 0) first = j;
    joints.push({ name: local(T.mjOBJ_JOINT.value, j, j - first), adr, n: QPOS_SIZE[model.jnt_type[j]] ?? 1 });
  }
  for (let a = at.ctrl; a < at.ctrl + robot.visual.dims.nu; a++) {
    ctrls.push({ name: local(T.mjOBJ_ACTUATOR.value, a, a - at.ctrl), a });
  }
  const b0 = robot.bodyOffset;
  for (let b = b0; b < b0 + robot.visual.inertia.nbody - 1; b++) {
    const m = model.body_mocapid[b];
    if (m >= 0) mocaps.push({ name: local(T.mjOBJ_BODY.value, b, b - b0), m });
  }
  return { joints, ctrls, mocaps };
}

// A robot's state as an item's `state`.
export function captureState(mujoco, model, data, robot, at) {
  const { joints, ctrls, mocaps } = robotTables(mujoco, model, robot, at);
  const qpos = {}, ctrl = {}, mocap = {};
  for (const { name, adr, n } of joints) {
    qpos[name] = n === 1 ? round(data.qpos[adr]) : Array.from(data.qpos.subarray(adr, adr + n), round);
  }
  for (const { name, a } of ctrls) ctrl[name] = round(data.ctrl[a]);
  for (const { name, m } of mocaps) {
    mocap[name] = { pos: Array.from(data.mocap_pos.subarray(3 * m, 3 * m + 3), round),
                    quat: Array.from(data.mocap_quat.subarray(4 * m, 4 * m + 4), round) };
  }
  return { qpos, ctrl, mocap };
}

// Writes an item's `state` into `data` (names the robot does not have, or
// of another size, are skipped); the robot comes to rest. Returns how many
// were skipped.
export function applyState(mujoco, model, data, robot, at, state = {}) {
  const { joints, ctrls, mocaps } = robotTables(mujoco, model, robot, at);
  let skipped = 0;
  const byName = (list) => new Map(list.map((x) => [x.name, x]));
  const J = byName(joints), C = byName(ctrls), M = byName(mocaps);
  for (const [name, v] of Object.entries(state.qpos ?? {})) {
    const j = J.get(name), vals = [v].flat();
    if (!j || vals.length !== j.n) { skipped++; continue; }
    data.qpos.set(vals, j.adr);
  }
  for (const [name, v] of Object.entries(state.ctrl ?? {})) {
    const c = C.get(name);
    if (c) data.ctrl[c.a] = v; else skipped++;
  }
  for (const [name, { pos, quat }] of Object.entries(state.mocap ?? {})) {
    const m = M.get(name);
    if (!m) { skipped++; continue; }
    data.mocap_pos.set(pos, 3 * m.m);
    data.mocap_quat.set(quat, 4 * m.m);
  }
  data.qvel.fill(0, at.qvel, at.qvel + robot.visual.dims.nv);
  data.act.fill(0, at.act, at.act + robot.visual.dims.na);
  return skipped;
}

// A random (version 4) UUID. crypto.randomUUID is only there in a secure
// context (https, localhost); a page opened over http by another address
// makes one from getRandomValues, which is there everywhere.
export function newId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = [...b].map((x) => x.toString(16).padStart(2, '0')).join('');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

// The main view of `app` as a stage's `view`, and back.
export const viewOf = (orbit) => ({ center: orbit.center.map(round), dist: round(orbit.dist), azim: round(orbit.azim), elev: round(orbit.elev) });
export function setView(orbit, view) {
  if (!view) return;
  orbit.track(null);
  orbit.focusing = null;
  orbit.center = [...view.center];
  Object.assign(orbit, { dist: view.dist, azim: view.azim, elev: view.elev });
}

// One line per stage for a list: {id, name, saved, count, author, thumb (URL
// or null), where ('mine' | 'community'), deletable, store}.
const summary = (stage, thumb, store) => ({
  id: stage.id, name: stage.name, saved: stage.saved, count: stage.count ?? stage.items?.length ?? 0,
  author: stage.author ?? '', thumb, where: store.where, deletable: store.deletable, store,
});

export const blobToDataUrl = (blob) => new Promise((resolve, reject) => {
  const fr = new FileReader();
  fr.onload = () => resolve(fr.result);
  fr.onerror = () => reject(fr.error);
  fr.readAsDataURL(blob);
});

// A bundle's files as read from a stage folder at `base` (a URL ending in /).
async function readFolder(base, id) {
  const get = (f) => fetch(new URL(f, base)).then((r) => (r.ok ? r : null)).catch(() => null);
  const stage = await (await get('stage.json'))?.json();
  if (!stage) throw new Error(`no stage ${id}`);
  const files = {};
  for (const f of FILES) files[f] = (await (await get(f))?.json().catch(() => null)) ?? {};
  const thumb = await get('thumbnail.jpg');
  return { stage: { ...stage, id }, files, thumbnail: thumb ? await blobToDataUrl(await thumb.blob()) : null };
}

// Checks a bundle read from a file (Import); throws if it is not one.
export function checkBundle(b) {
  if (!b?.stage || !Array.isArray(b.stage.items)) throw new Error('not a stage file');
  if (b.stage.version > STAGE_VERSION) throw new Error(`stage version ${b.stage.version} is newer than this page (${STAGE_VERSION})`);
  for (const it of b.stage.items) {
    if (typeof it.rel !== 'string' || !Array.isArray(it.pos)) throw new Error('not a stage file: an item without rel or pos');
  }
  return { stage: b.stage, files: Object.fromEntries(FILES.map((f) => [f, b.files?.[f] ?? {}])), thumbnail: b.thumbnail ?? null };
}

// Stages in this browser (IndexedDB), a record per stage: a bundle.
export class BrowserStages {
  where = 'mine';
  deletable = true;

  db() {
    this.opened ??= new Promise((resolve, reject) => {
      const req = indexedDB.open('robot-lobby', 1);
      req.onupgradeneeded = () => req.result.createObjectStore('stages', { keyPath: 'stage.id' });
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return this.opened;
  }

  async run(mode, fn) {
    const db = await this.db();
    return new Promise((resolve, reject) => {
      const req = fn(db.transaction('stages', mode).objectStore('stages'));
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  async list() {
    const all = await this.run('readonly', (s) => s.getAll());
    return all.map((b) => summary(b.stage, b.thumbnail, this));
  }

  get(id) { return this.run('readonly', (s) => s.get(id)).then((b) => b ?? Promise.reject(new Error(`no stage ${id}`))); }
  put(bundle) { return this.run('readwrite', (s) => s.put(bundle)).then(() => bundle.stage.id); }
  remove(id) { return this.run('readwrite', (s) => s.delete(id)); }
}

// The published stages of the library next to the page (local mode):
// mjcf_assets/saved_stage/ (public/json/stages.json), read-only.
export class LibraryStages {
  where = 'community';
  deletable = false;

  async list() {
    const index = await fetch('json/stages.json').then((r) => (r.ok ? r.json() : [])).catch(() => []);
    return index.map((s) => summary(s, `${STAGE_DIR}/${s.id}/thumbnail.jpg`, this));
  }

  get(id) { return readFolder(new URL(`${STAGE_DIR}/${id}/`, location.href), id); }
}

// The server's published stages (network mode): saved_stage/ of its shared
// library. The server has the world: `capture(extra)` gets it as a new
// bundle (not kept there) with what only the page knows (name, thumbnail,
// view, cameras, on), see server.py api_capture_stage; `publish(bundle,
// author)` writes one (author: {name}, optional) and resolves with its
// summary ({id, …}).
export class ServerStages {
  where = 'community';
  deletable = false;  // for now (the server has DELETE /api/stages/<id>)

  async request(path, init) {
    const res = await fetch(serverUrl(path), init);
    if (!res.ok) throw new Error(await res.text() || res.statusText);
    return res.json();
  }

  async list() {
    const all = await this.request('api/stages');
    return all.map((s) => summary(s, s.thumb ? dataUrl(`${STAGE_DIR}/${s.id}/thumbnail.jpg`) : null, this));  // {id, name, saved, count, thumb}
  }

  get(id) { return this.request(`api/stages/${id}`); }
  capture(extra) { return this.request('api/stages/capture', { method: 'POST', body: JSON.stringify(extra) }); }
  publish(bundle, author) { return this.request('api/stages/publish', { method: 'POST', body: JSON.stringify({ bundle, author }) }); }
  remove(id) { return this.request(`api/stages/${id}`, { method: 'DELETE' }); }
}

const AUTHOR_KEY = 'mujoco-filament/author';  // localStorage: {name} last published with

const PARTS_KEY = 'mujoco-filament/stage-parts';  // localStorage: the parts last chosen, {key: bool}

// What a stage may keep besides its models and where they stand, as the
// Save and Publish dialogs offer it: [key, label, tooltip]. A stage without
// a part loads without it: the physics settings, the view, the camera and
// controller setups stay as they are or come from the model, the models
// stand in their spawn pose.
export const STAGE_PARTS = [
  ['state', 'Joint state', 'joint positions, actuator controls and mocap bodies as they are now (else the models\' spawn pose)'],
  ['use', 'Controllers on', 'which controllers are switched on (policy, IK, mobile base)'],
  ['ik', 'IK and base setups', 'IK chains and mobile base setups edited in the Teleop panel'],
  ['cameras', 'Cameras', 'camera setups and which robot cameras are switched on'],
  ['physics', 'Physics', 'the Physics panel settings'],
  ['view', 'View', 'the main camera\'s orbit'],
];

// A bundle without the parts `parts` leaves out ({key: false}).
export function withParts(bundle, parts) {
  const off = (k) => parts[k] === false, stage = { ...bundle.stage }, files = { ...bundle.files };
  if (off('physics')) delete stage.physics;
  if (off('view')) delete stage.view;
  if (off('ik')) { delete files['ik.json']; delete files['base.json']; }
  if (off('cameras')) delete files['cameras.json'];
  stage.items = (stage.items ?? []).map((it) => {
    it = { ...it };
    if (off('state')) delete it.state;
    if (off('use')) delete it.use;
    if (off('cameras')) delete it.cameras;
    return it;
  });
  return { ...bundle, stage, files };
}

// A modal form (the Save stage dialog's look): `fields` the HTML between the
// title and the hint; resolves to read(form elements) on OK, null on Cancel.
function formDialog({ title, fields, hint, ok, init, read }) {
  const dlg = document.createElement('dialog');
  dlg.className = 'publish';
  dlg.innerHTML = `<form method="dialog">
    <h3></h3>
    ${fields}
    <p class="hint"></p>
    <div class="buttons"><button value="cancel" formnovalidate>Cancel</button><button value="ok" class="primary"></button></div>
  </form>`;
  dlg.querySelector('h3').textContent = title;
  dlg.querySelector('.hint').textContent = hint;
  dlg.querySelector('button.primary').textContent = ok;
  const f = dlg.querySelector('form').elements;
  init?.(f);
  document.body.append(dlg);
  return new Promise((resolve) => {
    dlg.onclose = () => {
      dlg.remove();
      resolve(dlg.returnValue === 'ok' ? read(f) : null);
    };
    dlg.showModal();
  });
}

// Asks for a name (the Save stage dialog without the stage's parts): the
// name, or null (cancelled). E.g. the Behavior panel's Save.
export function askName(title, name, { hint = '', ok = 'Save' } = {}) {
  return formDialog({
    title, hint, ok,
    fields: '<label>Name <input name="name" required maxlength="200"></label>',
    init: (f) => { f.name.value = name ?? ''; },
    read: (f) => f.name.value.trim() || null,
  });
}

// Asks what to save or publish a stage as (`publish`: with an author):
// {name, parts: {key: bool}, author: {name}}, or null (cancelled).
// The author and the parts are remembered for next time.
export function askStage(name, { publish = false } = {}) {
  let saved = {}, parts = {};
  try { saved = JSON.parse(localStorage.getItem(AUTHOR_KEY)) ?? {}; } catch { /* none */ }
  try { parts = JSON.parse(localStorage.getItem(PARTS_KEY)) ?? {}; } catch { /* none */ }
  return formDialog({
    title: `${publish ? 'Publish' : 'Save'} stage`,
    ok: publish ? 'Publish' : 'Save',
    hint: publish
      ? 'Published stages are listed for everyone on this server (Community), with this name as their author.'
      : 'Saved in this browser (My stages). What is left out stays as it is when the stage is loaded.',
    fields: `<label>Name <input name="name" required maxlength="200"></label>
    ${publish ? '<label>Author <input name="author" maxlength="100" placeholder="optional" autocomplete="name"></label>' : ''}
    <fieldset class="parts"><legend>Keep, besides the models and where they stand</legend>
      ${STAGE_PARTS.map(([k, label, title]) => `<label title="${title}"><input type="checkbox" name="part-${k}"> ${label}</label>`).join('')}
    </fieldset>`,
    init: (f) => {
      f.name.value = name ?? '';
      if (publish) f.author.value = saved.name ?? '';
      for (const [k] of STAGE_PARTS) f[`part-${k}`].checked = parts[k] !== false;
    },
    read: (f) => {
      const chosen = Object.fromEntries(STAGE_PARTS.map(([k]) => [k, f[`part-${k}`].checked]));
      try { localStorage.setItem(PARTS_KEY, JSON.stringify(chosen)); } catch { /* no storage */ }
      let author = null;
      if (publish) {
        author = { name: f.author.value.trim() };
        try { localStorage.setItem(AUTHOR_KEY, JSON.stringify(author)); } catch { /* no storage */ }
      }
      return { name: f.name.value.trim() || 'Stage', parts: chosen, author };
    },
  });
}

// Saves a bundle as a file (Download).
export function downloadBundle(bundle) {
  const name = (bundle.stage.name || 'stage').replace(/[^\w.-]+/g, '_');
  const url = URL.createObjectURL(new Blob([JSON.stringify(bundle)], { type: 'application/json' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: `${name}.stage.json` });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// A bundle from a file the user picks (Import); null if none is picked.
export function pickBundle() {
  return new Promise((resolve, reject) => {
    const input = Object.assign(document.createElement('input'), { type: 'file', accept: '.json,application/json' });
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) { resolve(null); return; }
      try { resolve(checkBundle(JSON.parse(await file.text()))); } catch (err) { reject(err); }
    };
    input.oncancel = () => resolve(null);
    input.click();
  });
}

const SHOWN = 20;  // stages listed, the newest

// A list of stages in the Stage panel (Mine, Community): a row each with
// its thumbnail, name, author, when and how many items, and Load / Publish /
// Download / Link / Delete. `actions`: {load(entry), publish(entry) (left
// out: not offered), download(entry), link(entry): a URL that opens it (null:
// none, it is in this browser only), remove(entry)}; `empty`: the text of
// an empty list.
export class StageList {
  constructor(el, actions, empty = 'No saved stages yet.') {
    this.el = el;
    this.actions = actions;
    this.empty = empty;
    this.entries = [];
  }

  // `entries` newest first: the SHOWN newest of them are listed.
  show(entries) {
    this.entries = entries;
    const list = this.el.querySelector('.stage-items');
    list.replaceChildren();
    const shown = entries.slice(0, SHOWN);
    this.el.querySelector('.stage-count').textContent = entries.length > SHOWN ? `${SHOWN} of ${entries.length}` : entries.length;
    if (!entries.length) {
      list.append(Object.assign(document.createElement('div'), { className: 'hint', textContent: this.empty }));
      return;
    }
    for (const e of shown) {
      const row = document.createElement('div');
      row.className = 'stage-item';
      const img = document.createElement(e.thumb ? 'img' : 'div');
      img.className = 'thumb';
      if (e.thumb) { img.src = e.thumb; img.alt = ''; img.loading = 'lazy'; }
      const info = document.createElement('div');
      info.className = 'info';
      const name = Object.assign(document.createElement('div'), { className: 'name', textContent: e.name || '(unnamed)', title: `${e.name} · ${e.id}` });
      const when = e.saved ? new Date(e.saved).toLocaleString([], { year: 'numeric', month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
      const metaText = `${e.count} item${e.count === 1 ? '' : 's'} · ${when}${e.author ? ` · by ${e.author}` : ''}`;
      const meta = Object.assign(document.createElement('div'), { className: 'meta', textContent: metaText, title: metaText });
      const buttons = document.createElement('div');
      buttons.className = 'buttons';
      const button = (text, title, fn) => {
        const b = Object.assign(document.createElement('button'), { textContent: text, title });
        b.onclick = fn;
        buttons.append(b);
        return b;
      };
      button('Load', 'Replace what is spawned with this stage', () => this.actions.load(e));
      if (this.actions.publish) button('Publish', 'Share this stage with everyone on this server (Community)', () => this.actions.publish(e));
      button('Download', 'Save this stage as a file (Import takes it, in either mode)', () => this.actions.download(e));
      const link = this.actions.link(e);
      if (link) {
        const b = button('Link', 'Copy a link that opens this stage', async () => {
          try { await navigator.clipboard.writeText(link); b.textContent = 'Copied'; } catch { prompt('Link to this stage:', link); }
          setTimeout(() => { b.textContent = 'Link'; }, 1500);
        });
      }
      if (e.deletable) button('✕', 'Delete this stage', () => this.actions.remove(e)).className = 'remove';
      info.append(name, meta, buttons);
      row.append(img, info);
      list.append(row);
    }
  }
}
