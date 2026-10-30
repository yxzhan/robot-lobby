// Who is here and who drives what, in sight without opening a panel (shared
// world only; control.js has the data):
// Everyone is shown by their mark (control.js: an icon of their kind of
// device, PC, Mobile, VR or Script, in their color).
// - a tag over every held robot: its holder's mark and name ("you" for
//   this page's), fading once the holder has left it alone (it is about to
//   be let go, or taken over);
// - the people row under the toolbar's status: everyone connected, this
//   page's own name first (click it to rename), a tooltip with each one's
//   device and robots;
// - notices for a moment at the bottom: someone took over a robot this
//   page drove, someone drives the robot this page's Teleop panel shows,
//   people joining, leaving or renaming.

import { mark } from './control.js';

const NOTICE_MS = 4000;
const TAGS_KEY = 'robot-lobby/name-tags';  // localStorage: '0' hides the tags (the Share panel's switch)

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};
const robotName = (r) => `${r.prefix.slice(0, -1)} · ${r.label}`;

export class Presence {
  constructor(app, control) {
    this.app = app;
    this.control = control;
    this.layer = el('div', { id: 'tags' });
    this.notices = el('div', { id: 'notices' });
    document.getElementById('canvas').after(this.layer);  // over the view, under the panels and toolbar
    document.body.append(this.notices);
    this.row = el('div', { id: 'people' });
    document.getElementById('ui').append(this.row);
    this.tags = new Map();   // robot netId -> its tag
    this.seenClients = null; // id -> {name, here}: what the last clients message said
    this.seenHolds = null;   // robot netId -> holder id
    this.editing = false;
    try { this.showTags = localStorage.getItem(TAGS_KEY) !== '0'; } catch { this.showTags = true; }
    this.layer.hidden = !this.showTags;
  }

  // The tags over held robots on or off (saved in this browser).
  setTags(on) {
    this.showTags = on;
    try { localStorage.setItem(TAGS_KEY, on ? '1' : '0'); } catch { /* no storage */ }
    this.layer.hidden = !on;
  }

  robots() {
    return this.app.teleop?.controllable() ?? [];
  }

  // control.js's clients or holds changed ('idle': only the idle times).
  changed(what) {
    if (what === 'clients') this.clientNotices();
    if (what === 'holds') this.holdNotices();
    if (!this.editing) this.renderRow();
  }

  clientNotices() {
    const c = this.control, now = new Map([...c.clients].map(([id, x]) => [id, { name: x.name, here: x.here, kind: x.kind, color: x.color }]));
    const before = this.seenClients;
    this.seenClients = now;
    if (!before || !c.you) return;  // what was there on joining is no news
    for (const [id, x] of now) {
      if (id === c.you) continue;
      const was = before.get(id), who = c.clients.get(id);
      if (!was && x.here) this.notice(`${x.name} joined${who.device ? ` (${who.device})` : ''}`, who);
      else if (was?.here && !x.here) this.notice(`${x.name} left`, who);
      else if (was && was.name !== x.name) this.notice(`${was.name} is now ${x.name}`, who);
    }
    for (const [id, was] of before) if (!now.has(id) && was.here && id !== c.you) this.notice(`${was.name} left`, was);
  }

  holdNotices() {
    const c = this.control, now = new Map([...c.holds].map(([id, h]) => [id, h.client]));
    const before = this.seenHolds;
    this.seenHolds = now;
    if (!before || !c.you) return;
    const active = this.app.teleop?.active;
    for (const r of this.robots()) {
      const was = before.get(r.netId), is = now.get(r.netId);
      if (is === was || !is) continue;
      const who = c.clients.get(is);
      if (!who || is === c.you) continue;
      if (was === c.you) this.notice(`${who.name} took over ${robotName(r)}`, who, true);
      else if (r === active) this.notice(`${who.name} drives ${robotName(r)} now`, who);
    }
  }

  // `who`: the client it is about (its mark).
  notice(text, who, warn = false) {
    const n = el('div', { className: 'notice' + (warn ? ' warn' : '') }, mark(who), el('span', { textContent: text }));
    this.notices.append(n);
    while (this.notices.childElementCount > 4) this.notices.firstChild.remove();
    setTimeout(() => { n.classList.add('gone'); setTimeout(() => n.remove(), 400); }, NOTICE_MS);
  }

  // The people row: this page first (click to rename), then the others.
  renderRow() {
    const c = this.control, robots = this.robots();
    const drives = (id) => robots.filter((r) => c.holder(r)?.id === id).map(robotName);
    const people = [...c.clients.values()].filter((x) => x.here)
      .sort((a, b) => (b.id === c.you) - (a.id === c.you));
    this.row.replaceChildren(...people.map((x) => {
      const mine = x.id === c.you, d = drives(x.id);
      const chip = el('span', { className: 'person' + (mine ? ' you' : ''),
        title: [mine ? 'You (click to rename)' : x.name, x.device, d.length ? `drives ${d.join(', ')}` : ''].filter(Boolean).join('\n') },
      mark(x), el('span', { className: 'name', textContent: mine ? `${x.name} (you)` : x.name }));
      if (mine) chip.onclick = () => this.rename(chip, x.name);
      return chip;
    }));
  }

  rename(chip, name) {
    this.editing = true;
    const input = el('input', { type: 'text', value: this.control.name || name, maxLength: 40, placeholder: name });
    chip.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = (save) => {
      if (done) return;
      done = true;
      this.editing = false;
      if (save && input.value.trim() && input.value.trim() !== name) this.control.rename(input.value);
      this.renderRow();
    };
    input.onkeydown = (e) => {
      e.stopPropagation();  // typing is not driving
      if (e.key === 'Enter') finish(true);
      if (e.key === 'Escape') finish(false);
    };
    input.onblur = () => finish(true);
  }

  // Every frame: the tags over the held robots, where their tops are now.
  update() {
    if (!this.showTags) return;
    const { control: c, app } = this, data = app.data;
    const seen = new Set();
    for (const r of data ? this.robots() : []) {
      const h = c.holder(r);
      if (!h) continue;
      const n = r.visual?.inertia?.nbody ?? 1;
      let top = -Infinity;
      for (let b = r.bodyOffset; b < r.bodyOffset + n - 1; b++) top = Math.max(top, data.xpos[3 * b + 2]);
      const root = r.bodyOffset;
      const at = app.toScreen([data.xpos[3 * root], data.xpos[3 * root + 1], top + 0.2]);
      let tag = this.tags.get(r.netId);
      if (!at) { if (tag) tag.hidden = true; continue; }
      if (!tag) {
        tag = el('div', { className: 'tag' }, mark(h), el('span'));
        this.layer.append(tag);
        this.tags.set(r.netId, tag);
      }
      seen.add(r.netId);
      const text = c.mine(r) ? 'you' : h.name;
      if (tag.lastChild.textContent !== text) tag.lastChild.textContent = text;
      if (tag.dataset.who !== `${h.kind}${h.color}`) {
        tag.dataset.who = `${h.kind}${h.color}`;
        tag.firstChild.replaceWith(mark(h));
      }
      tag.hidden = false;
      tag.classList.toggle('idle', c.idle(r.netId) >= 2 || !h.here);
      tag.style.transform = `translate(${Math.round(at.x)}px, ${Math.round(at.y)}px) translate(-50%, -100%)`;
    }
    for (const [id, tag] of this.tags) if (!seen.has(id)) { tag.remove(); this.tags.delete(id); }
  }
}
