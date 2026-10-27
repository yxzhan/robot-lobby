// The Behavior panel: a behaviour tree (bt.js) for the robots, written in
// its text form (or pasted as BehaviorTree.CPP XML), run and stopped here,
// the tree shown below with each node's status as it runs. The text is kept
// in this browser; Save keeps it under a name there too (My trees), next to
// the examples (public/behaviors: index.json lists the files). window.bt
// does the same from the devtools console:
//   bt.run('Move all forward=1')  bt.stop()  bt.status()

import { IDLE, RUNNING, Runner, nodes, parse } from './bt.js';

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

const TEXT = 'mujoco-filament/behavior-tree';    // the editor's text
const SAVED = 'mujoco-filament/behavior-trees';  // {name: text}, the user's trees
const EXAMPLES = new URL('../behaviors/', import.meta.url);  // index.json: [{name, file}]

const DEFAULT = 'Move all forward=0.5';

const readSaved = () => {
  try { return JSON.parse(localStorage.getItem(SAVED)) ?? {}; } catch { return {}; }
};

export class BehaviorPanel {
  constructor(app, root, goals) {
    this.app = app;
    this.goals = goals;
    this.body = root.querySelector('.body');
    this.runner = null;
    this.rows = new Map();  // node -> its row in the tree view
    this.examples = {};     // name -> text, from public/behaviors
    let text = DEFAULT;
    try { text = localStorage.getItem(TEXT) ?? DEFAULT; } catch { /* no storage */ }

    // Open: an example or a saved tree into the editor (its text replaced).
    this.picker = el('select', { title: 'Put an example or one of your saved trees in the editor (the text there is replaced)' });
    this.picker.onchange = () => {
      const [kind, name] = [this.picker.value.slice(0, 3), this.picker.value.slice(3)];
      if (kind === 'ex:') this.setText(this.examples[name]);
      if (kind === 'my:') this.setText(readSaved()[name]);
      this.picker.blur();
      this.update();
    };
    const save = el('button', { type: 'button', textContent: 'Save', title: 'Keep the editor\'s tree in this browser under a name (My trees)' });
    save.onclick = () => this.saveAs();
    this.deleteButton = el('button', { type: 'button', textContent: 'Delete', title: 'Forget the saved tree picked on the left' });
    this.deleteButton.onclick = () => this.remove();
    this.editor = el('textarea', { value: text, spellcheck: false, rows: 12,
      title: 'One node a line, children indented under it; or BehaviorTree.CPP XML. Ctrl+Enter runs it' });
    this.editor.onkeydown = (e) => {
      e.stopPropagation();  // keys do not drive the robot
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); this.run(); }
      if (e.key === 'Tab') {  // indents instead of leaving the box
        e.preventDefault();
        this.editor.setRangeText('  ', this.editor.selectionStart, this.editor.selectionEnd, 'end');
      }
    };
    this.editor.oninput = () => this.save();
    this.runButton = el('button', { type: 'button', className: 'primary', textContent: 'Run', title: 'Start the tree from the top (Ctrl+Enter)' });
    this.runButton.onclick = () => this.run();
    this.stopButton = el('button', { type: 'button', textContent: 'Stop', title: 'Stop the tree; the robots it drives stop where they are' });
    this.stopButton.onclick = () => this.stop();
    this.note = el('span', { className: 'note' });
    this.tree = el('div', { className: 'bt-tree' });
    this.body.append(
      el('div', { className: 'bar' }, this.picker, save, this.deleteButton),
      this.editor,
      el('div', { className: 'bar' }, this.runButton, this.stopButton, this.note),
      this.tree);
    this.listTrees();
    fetch(new URL('index.json', EXAMPLES)).then((r) => r.json())
      .then((list) => Promise.all(list.map(async ({ name, file }) => [name, await fetch(new URL(file, EXAMPLES)).then((r) => r.text())])))
      .then((entries) => { this.examples = Object.fromEntries(entries); this.listTrees(); })
      .catch((err) => console.warn('behaviour tree examples:', err));

    window.bt = {
      run: (t) => { if (t != null) this.setText(t); return this.run(); },
      stop: () => this.stop(),
      status: () => this.runner && {
        status: this.runner.status, error: this.runner.error, time: this.runner.time,
        nodes: nodes(this.runner.root).map((n) => ({ node: n.text, status: n.status, info: n.info ?? '' })),
      },
    };
  }

  setText(t) {
    this.editor.value = t;
    this.save();
  }

  save() {
    try { localStorage.setItem(TEXT, this.editor.value); } catch { /* no storage */ }
  }

  // The picker's entries: the examples, then the saved trees; `pick` selected.
  listTrees(pick = this.picker.value) {
    const group = (label, kind, names) => {
      const g = el('optgroup', { label });
      for (const n of names) g.append(new Option(n, kind + n));
      return g;
    };
    const saved = Object.keys(readSaved()).sort((a, b) => a.localeCompare(b));
    this.picker.replaceChildren(new Option('Open…', ''), group('Examples', 'ex:', Object.keys(this.examples)),
      ...(saved.length ? [group('My trees', 'my:', saved)] : []));
    this.picker.value = [...this.picker.options].some((o) => o.value === pick) ? pick : '';
    this.update();
  }

  // Saves the editor's tree under a name asked for (the picked saved tree's by default).
  saveAs() {
    const picked = this.picker.value.startsWith('my:') ? this.picker.value.slice(3) : '';
    const name = prompt('Save the tree as', picked)?.trim();
    if (!name) return;
    const saved = readSaved();
    if (name !== picked && saved[name] != null && !confirm(`Replace your tree "${name}"?`)) return;
    saved[name] = this.editor.value;
    try {
      localStorage.setItem(SAVED, JSON.stringify(saved));
    } catch (err) {
      this.note.textContent = `not saved: ${err.message ?? err}`;
      return;
    }
    this.listTrees(`my:${name}`);
  }

  remove() {
    const name = this.picker.value.slice(3);
    if (!this.picker.value.startsWith('my:') || !confirm(`Delete your tree "${name}"?`)) return;
    const saved = readSaved();
    delete saved[name];
    try { localStorage.setItem(SAVED, JSON.stringify(saved)); } catch { /* no storage */ }
    this.listTrees('');
  }

  // Starts the editor's tree (stopping one that runs); false if it does not parse.
  run() {
    this.runner?.stop();
    this.runner = null;
    let root;
    try {
      root = parse(this.editor.value);
    } catch (err) {
      this.error = err.message;
      this.showTree(null);
      this.update();
      return false;
    }
    this.error = null;
    this.showTree(root);
    this.runner = new Runner(root, { app: this.app, goals: this.goals, onTick: () => this.update() });
    return true;
  }

  stop() {
    this.runner?.stop();
    this.update();
  }

  // The tree view: a row a node, indented, its status as a coloured dot.
  showTree(root) {
    this.rows.clear();
    this.tree.replaceChildren();
    if (!root) return;
    const add = (n, depth) => {
      const info = el('span', { className: 'info' });
      const row = el('div', { className: 'node', title: n.where }, el('span', { className: 'dot' }), el('span', { textContent: n.text }), info);
      row.style.paddingLeft = `${depth * 12}px`;
      this.rows.set(n, { row, info });
      this.tree.append(row);
      n.children.forEach((c) => add(c, depth + 1));
    };
    add(root, 0);
  }

  update() {
    const r = this.runner;
    for (const [n, { row, info }] of this.rows) {
      row.dataset.status = n.status;
      info.textContent = n.status === IDLE || !n.info ? '' : n.info;
    }
    const running = r?.status === RUNNING;
    this.deleteButton.disabled = !this.picker.value.startsWith('my:');
    this.stopButton.disabled = !running;
    this.runButton.textContent = running ? 'Restart' : 'Run';
    this.note.classList.toggle('bad', !!(this.error || r?.error));
    this.note.textContent = this.error ?? (r?.error ? `stopped: ${r.error}`
      : !r ? '' : running ? `running ${(r.time ?? 0).toFixed(1)} s${this.app.paused ? ' (paused)' : ''}`
      : r.status === IDLE ? 'stopped' : `${r.status} after ${(r.time ?? 0).toFixed(1)} s`);
  }
}
