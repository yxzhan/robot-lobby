// The asset panel: the models of the library index (public/assets.json,
// written by scripts/vendor.mjs from mjcf_assets/) under three tabs,
// environments / robots / objects, with a name filter; within a tab, models
// are listed under foldable group headers (their source, or menagerie's
// README sections), in the index's order.
// An item is spawned by dragging it onto the floor: while dragging, a marker
// on the floor shows the spawn point and heading, which starts out along the
// world +x axis and turns in 15° steps with the mouse wheel or Q/E; Shift+wheel or
// R/F raise and lower it in 5 cm steps (the height is kept for the next
// drag); Esc cancels. The drag label shows the position and heading.

export const CATEGORIES = [['environment', 'Environments'], ['robot', 'Robots'], ['object', 'Objects']];

// The library index; `rel` is the file to spawn, relative to public/.
export async function loadAssets(index = 'assets.json') {
  try {
    const res = await fetch(index);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return (await res.json()).map((m) => ({
      name: m.name, rel: `${m.dir}/${m.main}`, lib: m.source, category: m.category, group: m.group ?? null,
      thumb: m.screenshot ? thumbPath(m) : null,  // written by scripts/thumbnails.py
    }));
  } catch (err) {
    console.warn(`no asset index (${index}): run node scripts/vendor.mjs`, err);
    return [];
  }
}

// Where scripts/thumbnails.py puts a model's thumbnail (same rule there).
export const thumbPath = (m) => `thumbs/${m.category}/${m.source}/${m.name}.webp`;

const STEP = Math.PI / 12, HEIGHT_STEP = 0.05, MAX_HEIGHT = 5;

export class AssetPanel {
  // `onDrop(asset, pos, yaw)` spawns; `canDrop()` says whether that is possible now.
  constructor(app, el, assets, { onDrop, canDrop = () => true }) {
    this.app = app;
    this.el = el;
    this.assets = assets;
    this.onDrop = onDrop;
    this.canDrop = canDrop;
    this.tab = assets.some((a) => a.category === 'robot') ? 'robot' : 'environment';
    this.drag = null;  // {asset, ghost, label, readout, pos, turn, x, y}
    this.height = 0;  // spawn height above the floor, kept between drags
    this.collapsed = new Set();  // "<tab>/<group>" headers folded in the list

    const tabs = el.querySelector('.tabs');
    for (const [cat, label] of CATEGORIES) {
      const b = document.createElement('button');
      b.dataset.tab = cat;
      b.innerHTML = `${label} <span class="count">${assets.filter((a) => a.category === cat).length}</span>`;
      b.onclick = () => { this.tab = cat; this.render(); };
      tabs.append(b);
    }
    this.filter = el.querySelector('input[type=search]');
    this.filter.oninput = () => this.render();
    this.list = el.querySelector('ul');
    this.list.addEventListener('pointerdown', (e) => this.begin(e));
    window.addEventListener('pointermove', (e) => this.move(e));
    window.addEventListener('pointerup', (e) => this.end(e));
    window.addEventListener('pointercancel', () => this.cancel());
    // Capture phase on window: runs before the camera's zoom and the WASD/QE
    // drive keys, which must not see these while dragging.
    window.addEventListener('wheel', (e) => {
      if (!this.drag) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      // Shift+wheel scrolls horizontally on some systems: deltaX then.
      const delta = Math.sign(e.deltaY || e.deltaX);
      if (e.shiftKey) this.raise(-delta * HEIGHT_STEP);
      else this.turn(delta * STEP);
    }, { capture: true, passive: false });
    window.addEventListener('keydown', (e) => {
      if (!this.drag) return;
      const k = e.key.toLowerCase();
      if (k === 'escape') this.cancel();
      else if (k === 'q' || k === 'e') this.turn(k === 'q' ? STEP : -STEP);
      else if (k === 'r' || k === 'f') this.raise(k === 'r' ? HEIGHT_STEP : -HEIGHT_STEP);
      else return;
      e.preventDefault();
      e.stopImmediatePropagation();
    }, { capture: true });
    this.render();
  }

  render() {
    for (const b of this.el.querySelectorAll('.tabs button')) b.classList.toggle('active', b.dataset.tab === this.tab);
    const q = this.filter.value.trim().toLowerCase();
    const items = this.assets.filter((a) => a.category === this.tab && a.name.toLowerCase().includes(q));
    // Group headers where a tab's assets have groups, in the order the groups
    // first appear (the library indexes are sorted by group); the rest is
    // "Other". Each header folds its group.
    const grouped = items.some((a) => a.group);
    const groups = new Map();
    for (const a of items) {
      const g = grouped ? a.group ?? 'Other' : null;
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(a);
    }
    const rows = [];
    for (const [g, list] of groups) {
      if (g !== null) {
        const key = `${this.tab}/${g}`, folded = this.collapsed.has(key) && !q;
        const h = document.createElement('li');
        h.className = 'group' + (folded ? ' folded' : '');
        h.innerHTML = '<span class="caret"></span><span class="title"></span> <span class="count"></span>';
        h.children[1].textContent = g;
        h.children[2].textContent = list.length;
        h.onclick = () => { this.collapsed.has(key) ? this.collapsed.delete(key) : this.collapsed.add(key); this.render(); };
        rows.push(h);
        if (folded) continue;
      }
      rows.push(...list.map((a) => this.item(a)));
    }
    this.list.replaceChildren(...rows);
    if (!items.length) {
      const li = document.createElement('li');
      li.className = 'empty';
      li.textContent = q ? 'No match.'
        : this.tab === 'object' ? 'No objects yet: put models in mjcf_assets/objects/<source>/ and re-run node scripts/vendor.mjs.'
        : 'Nothing here.';
      this.list.append(li);
    }
  }

  // A card: thumbnail (or initials), name, library.
  item(a) {
    const li = document.createElement('li');
    li.dataset.rel = a.rel;
    li.title = `${a.rel}\nDrag onto the floor to spawn`;
    li.innerHTML = '<div class="thumb"></div><span class="name"></span><span class="lib"></span>';
    const [thumb, name, lib] = li.children;
    thumb.dataset.initial = a.name.slice(0, 2).toUpperCase();  // placeholder without a thumbnail
    if (a.thumb) {
      const img = new Image();
      img.loading = 'lazy';
      img.draggable = false;
      img.alt = '';
      img.onerror = () => img.remove();
      img.src = a.thumb;
      thumb.append(img);
    }
    name.textContent = a.name;
    lib.textContent = a.lib;
    return li;
  }

  begin(e) {
    const li = e.target.closest('li[data-rel]');
    if (!li || e.button !== 0) return;
    e.preventDefault();  // no text selection / native drag
    const asset = this.assets.find((a) => a.rel === li.dataset.rel);
    const ghost = document.createElement('div');
    ghost.className = 'drag-ghost';
    ghost.innerHTML = '<div class="name"></div><div class="readout"></div>';
    ghost.firstChild.textContent = asset.name;
    document.body.append(ghost);
    this.drag = { asset, ghost, readout: ghost.lastChild, pos: null, turn: 0, x: e.clientX, y: e.clientY };
    li.classList.add('dragging');
    this.move(e);
  }

  move(e) {
    const d = this.drag;
    if (!d) return;
    d.x = e.clientX;
    d.y = e.clientY;
    d.ghost.style.transform = `translate(${d.x + 14}px, ${d.y + 10}px)`;
    this.pick();
  }

  // The spawn point under the pointer, on the plane at the spawn height (so
  // the marker stays under the pointer when raised).
  pick() {
    const d = this.drag;
    const overCanvas = document.elementFromPoint(d.x, d.y) === this.app.canvas;
    d.pos = overCanvas ? this.app.pickFloor(d.x, d.y, this.height) : null;
    d.ghost.classList.toggle('placing', !!d.pos);
    this.show();
  }

  raise(delta) {
    this.height = Math.min(MAX_HEIGHT, Math.max(0, Math.round((this.height + delta) / HEIGHT_STEP) * HEIGHT_STEP));
    this.pick();
  }

  // Heading of the asset's +x: the world +x axis plus the user's turns.
  yaw() {
    return this.drag.turn;
  }

  turn(delta) {
    this.drag.turn += delta;
    this.show();
  }

  show() {
    const d = this.drag;
    const yaw = d?.pos ? this.yaw() : 0;
    this.app.setSpawnMarker(d?.pos ?? null, yaw);
    if (!d) return;
    if (!d.pos) {
      d.readout.textContent = `z ${this.height.toFixed(2)} m · drop on the floor`;
      return;
    }
    const deg = ((yaw * 180 / Math.PI) % 360 + 540) % 360 - 180;  // [-180, 180)
    const [x, y, z] = d.pos.map((v) => v.toFixed(2));
    d.readout.textContent = `x ${x}  y ${y}  z ${z} m · yaw ${deg.toFixed(0)}°`;
  }

  end(e) {
    const d = this.drag;
    if (!d) return;
    this.move(e);
    const { asset, pos } = d;
    const yaw = pos ? this.yaw() : 0;
    this.cancel();
    if (pos && this.canDrop()) this.onDrop(asset, pos, yaw);
  }

  cancel() {
    if (!this.drag) return;
    this.drag.ghost.remove();
    this.list.querySelector('.dragging')?.classList.remove('dragging');
    this.drag = null;
    this.app.setSpawnMarker(null);
  }
}
