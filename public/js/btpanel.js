// The Behavior panel: a behaviour tree for the robots, in the format picked
// (a py_trees script, pybt.js, by default; bt.js's text form; or
// BehaviorTree.CPP XML), written in a code editor (bteditor.js), run and
// stopped here, the tree shown below with each node's status as it runs.
// The text and the format are kept in this browser; Save keeps them under
// a name there too (My trees), next to the examples (public/behaviors:
// index.json lists the files; .py ones are py_trees, .xml XML, the others
// the text form). window.bt does the same from the devtools console:
//   bt.run('Move all forward=1', 'text')  bt.stop()  bt.status()

import { IDLE, RUNNING, Runner, parse } from './bt.js';
import { BtEditor } from './bteditor.js';
import { PyRunner, checkPython, loadPython } from './pybt.js';
import { askName } from './stage.js';

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

// A 12×12 SVG icon of `paths` (filled with the text colour).
const icon = (paths) => {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 12 12');
  svg.setAttribute('aria-hidden', 'true');
  svg.innerHTML = paths;
  return svg;
};

const TEXT = 'mujoco-filament/behavior-tree';      // the editor's text
const FORMAT = 'mujoco-filament/behavior-format';  // and its format
const FOLDED = 'mujoco-filament/behavior-folded';  // '1': the editor folded away
const SAVED = 'mujoco-filament/behavior-trees';    // {name: {text, format}}, the user's trees
const EXAMPLES = new URL('../behaviors/', import.meta.url);  // index.json: [{name, file}]

// [format, label]: the first is the default.
const FORMATS = [['python', 'py_trees (Python)'], ['text', 'Text form'], ['xml', 'BehaviorTree.CPP XML']];
const formatOfFile = (file) => (/\.py$/.test(file) ? 'python' : /\.xml$/.test(file) ? 'xml' : 'text');

const DEFAULT = `import py_trees
from lobby import Move

root = Move("all", forward=0.5)
`;

// The saved trees, {name: {text, format}} (an older one is a text-form string).
const readSaved = () => {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(SAVED)) ?? {}; } catch { /* none */ }
  return Object.fromEntries(Object.entries(saved).map(([k, v]) => [k, typeof v === 'string' ? { text: v, format: 'text' } : v]));
};

export class BehaviorPanel {
  constructor(app, root, goals) {
    this.app = app;
    this.goals = goals;
    this.body = root.querySelector('.body');
    this.runner = null;
    this.rows = [];         // the tree view's rows: {row, dot, info}
    this.shape = '';        // what they show (rebuilt when it changes)
    this.py = null;         // Pyodide, once a Python tree needed it
    this.examples = {};     // name -> {text, format}, from public/behaviors
    let text = DEFAULT, format = FORMATS[0][0];
    try {
      const t = localStorage.getItem(TEXT);
      if (t != null) [text, format] = [t, localStorage.getItem(FORMAT) ?? 'text'];  // (kept before there were formats: the text form)
    } catch { /* no storage */ }
    if (!FORMATS.some(([f]) => f === format)) format = FORMATS[0][0];
    this.format = format;

    // What the editor's text is: it is highlighted, checked and run as that.
    this.formatPicker = el('select', { title: 'What the tree is written in: a py_trees script (Python), the text form, or BehaviorTree.CPP XML' },
      ...FORMATS.map(([f, label]) => new Option(label, f, false, f === format)));
    this.formatPicker.onchange = () => {
      this.setFormat(this.formatPicker.value);
      this.formatPicker.blur();
    };

    // Open: an example or a saved tree into the editor (its text replaced).
    this.picker = el('select', { title: 'Put an example or one of your saved trees in the editor (the text there is replaced)' });
    this.picker.onchange = () => {
      const [kind, name] = [this.picker.value.slice(0, 3), this.picker.value.slice(3)];
      const tree = kind === 'ex:' ? this.examples[name] : kind === 'my:' ? readSaved()[name] : null;
      if (tree) this.setText(tree.text, tree.format);
      this.picker.blur();
      this.update();
    };
    const save = el('button', { type: 'button', textContent: 'Save', title: 'Keep the editor\'s tree in this browser under a name (My trees)' });
    save.onclick = () => this.saveAs();
    this.deleteButton = el('button', { type: 'button', textContent: 'Delete', title: 'Forget the saved tree picked on the left' });
    this.deleteButton.onclick = () => this.remove();
    this.runButton = el('button', { type: 'button', className: 'primary', title: 'Start the tree from the top (Ctrl+Enter)' });
    this.runLabel = el('span', { textContent: 'Run' });
    this.runButton.append(icon('<path d="M2 1.2v9.6a.8.8 0 0 0 1.2.7l8-4.8a.8.8 0 0 0 0-1.4l-8-4.8A.8.8 0 0 0 2 1.2z"/>'), this.runLabel);
    this.runButton.onclick = () => this.run();
    this.stopButton = el('button', { type: 'button', title: 'Stop the tree; the robots it drives stop where they are' });
    this.stopButton.append(icon('<rect x="1.5" y="1.5" width="9" height="9" rx="1.5"/>'), 'Stop');
    this.stopButton.onclick = () => this.stop();
    this.note = el('span', { className: 'note' });
    this.tree = el('div', { className: 'bt-tree' });
    const editorBox = el('div', { className: 'bt-editor' });
    // Folds the editor away (the tree view and Run / Stop stay), kept in this browser.
    const fold = el('button', { type: 'button', className: 'bt-fold' });
    const setFolded = (folded) => {
      editorBox.hidden = folded;
      fold.textContent = folded ? '▸' : '▾';
      fold.title = folded ? 'Show the editor' : 'Fold the editor away (the tree and Run / Stop stay)';
      fold.setAttribute('aria-expanded', String(!folded));
      try { localStorage.setItem(FOLDED, folded ? '1' : ''); } catch { /* no storage */ }
    };
    fold.onclick = () => { setFolded(!editorBox.hidden); fold.blur(); };
    let folded = false;
    try { folded = localStorage.getItem(FOLDED) === '1'; } catch { /* no storage */ }
    setFolded(folded);
    this.body.append(
      el('div', { className: 'bar' }, fold, this.formatPicker, this.picker, save, this.deleteButton),
      editorBox,
      el('div', { className: 'bar' }, this.runButton, this.stopButton, this.note),
      this.tree);
    this.editor = new BtEditor(editorBox, {
      value: text,
      format,
      onChange: () => { this.save(); this.wantPython(); },
      onRun: () => this.run(),
      lint: (t) => this.lint(t),
    });
    this.wantPython();
    this.listTrees();
    fetch(new URL('index.json', EXAMPLES)).then((r) => r.json())
      .then((list) => Promise.all(list.map(async ({ name, file }) => [name, {
        text: await fetch(new URL(file, EXAMPLES)).then((r) => r.text()), format: formatOfFile(file) }])))
      .then((entries) => { this.examples = Object.fromEntries(entries); this.listTrees(); })
      .catch((err) => console.warn('behaviour tree examples:', err));

    window.bt = {
      // confirm(what): asked before the tree removes or resets (the Agent's trees).
      run: (t, format = this.format, { confirm } = {}) => { if (t != null) this.setText(t, format); return this.run({ confirm }); },
      stop: () => this.stop(),
      status: () => this.runner && {
        status: this.runner.status, error: this.runner.error, time: this.runner.time,
        nodes: this.runner.rows().map((n) => ({ node: n.text, status: n.status, info: n.info })),
      },
    };
  }

  setText(t, format = this.format) {
    this.editor.value = t;
    this.setFormat(format);
  }

  setFormat(format) {
    if (format !== this.format) {
      this.format = format;
      this.editor.setFormat(format);
    }
    this.formatPicker.value = format;
    this.save();
    this.wantPython();
    this.editor.relint();
  }

  save() {
    try {
      localStorage.setItem(TEXT, this.editor.value);
      localStorage.setItem(FORMAT, this.format);
    } catch { /* no storage */ }
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
  async saveAs() {
    const picked = this.picker.value.startsWith('my:') ? this.picker.value.slice(3) : '';
    const name = await askName('Save behavior tree', picked, { hint: 'Saved in this browser (My trees).' });
    if (!name) return;
    const saved = readSaved();
    if (name !== picked && saved[name] != null && !confirm(`Replace your tree "${name}"?`)) return;
    saved[name] = { text: this.editor.value, format: this.format };
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

  // A Python script in the editor: Pyodide loads (in the background, once)
  // so that its syntax can be checked and it can run.
  wantPython() {
    if (this.py || this.format !== 'python') return;
    this.loadingPython ??= loadPython((msg) => { if (!this.runner) { this.note.textContent = msg; this.note.classList.remove('bad'); } })
      .then((py) => { this.py = py; this.editor.relint(); if (!this.runner) this.update(); return py; })
      .catch((err) => { this.loadingPython = null; this.note.textContent = `Python not loaded: ${err.message ?? err}`; throw err; });
    this.loadingPython.catch(() => {});
  }

  // The editor's errors: [{line, column?, message}].
  lint(text) {
    const out = [];
    if (this.format === 'python') {
      const e = checkPython(this.py, text);
      if (e) out.push(e);
    } else {
      try { parse(text, this.format); } catch (err) { out.push({ line: err.line ?? 1, message: err.message }); }
    }
    // The last run's error, while the text is the one that ran.
    const r = this.runner;
    if (r?.error && r.errorLine && r.source === text && !out.length) out.push({ line: r.errorLine, message: r.error });
    return out;
  }

  // Starts the editor's tree (stopping one that runs); false if it does not
  // parse (a Python one: once Python is loaded).
  async run({ confirm } = {}) {
    this.runner?.stop();
    this.runner = null;
    const source = this.editor.value, opts = { app: this.app, goals: this.goals, confirm, onTick: () => this.update() };
    this.error = null;
    if (this.format === 'python') {
      this.wantPython();
      this.update();
      let py;
      try { py = await this.loadingPython; } catch { return false; }
      if (this.editor.value !== source) return false;  // (edited meanwhile)
      this.runner = new PyRunner(py, source, opts);
    } else {
      let root;
      try {
        root = parse(source, this.format);
      } catch (err) {
        this.error = err.message;
        this.update();
        return false;
      }
      this.runner = new Runner(root, opts);
    }
    this.runner.source = source;
    this.update();
    this.editor.relint();
    return this.runner.status === RUNNING || !this.runner.error;
  }

  stop() {
    this.runner?.stop();
    this.update();
  }

  // The tree view: a row a node, indented, its status as a coloured dot.
  showTree(rows) {
    const shape = rows.map((r) => `${r.depth} ${r.text}`).join('\n');
    if (shape === this.shape) return;
    this.shape = shape;
    this.tree.replaceChildren();
    this.rows = rows.map((r) => {
      const info = el('span', { className: 'info' });
      const row = el('div', { className: 'node', title: r.title }, el('span', { className: 'dot' }), el('span', { textContent: r.text }), info);
      row.style.paddingLeft = `${r.depth * 12}px`;
      this.tree.append(row);
      return { row, info };
    });
  }

  update() {
    const r = this.runner;
    const rows = r?.rows() ?? [];
    this.showTree(rows);
    rows.forEach((x, i) => {
      this.rows[i].row.dataset.status = x.status;
      this.rows[i].info.textContent = x.info;
    });
    const running = r?.status === RUNNING;
    this.deleteButton.disabled = !this.picker.value.startsWith('my:');
    this.stopButton.disabled = !running;
    this.runLabel.textContent = running ? 'Restart' : 'Run';
    if (!r && !this.error && this.loadingPython && !this.py) return;  // (the loading message)
    const wasError = !!r?.error;
    this.note.classList.toggle('bad', !!(this.error || r?.error));
    this.note.textContent = this.error ?? (r?.error ? `stopped: ${r.error}`
      : !r ? '' : running ? `running ${(r.time ?? 0).toFixed(1)} s${this.app.paused ? ' (paused)' : ''}`
      : r.status === IDLE ? 'stopped' : `${r.status} after ${(r.time ?? 0).toFixed(1)} s`);
    if (wasError && !this.linted) { this.linted = true; this.editor.relint(); }
    if (!wasError) this.linted = false;
  }
}
