// The Behavior panel's editor: CodeMirror 6 (public/vendor/codemirror, a
// bundle of scripts/codemirror.mjs), loaded when the panel is built, a plain
// text box until then. Highlighting follows the panel's format (setFormat):
// a py_trees script, the text form of bt.js or BehaviorTree.CPP XML; the
// linter marks errors (`lint(text)` -> [{line, column?, message}]) in the
// gutter and under the text. Mod+Enter runs (`onRun`), Tab indents.

import { LEAVES, TYPES } from './bt.js';

const el = (tag, props = {}) => Object.assign(document.createElement(tag), props);

export class BtEditor {
  // format: 'python', 'text' or 'xml'.
  constructor(parent, { value, format, onChange, onRun, lint }) {
    this.format = format;
    this.onChange = onChange;
    this.onRun = onRun;
    this.lint = lint;
    this.box = el('textarea', { value, spellcheck: false, rows: 14, className: 'bt-plain',
      title: 'Ctrl+Enter runs it' });
    this.box.onkeydown = (e) => {
      e.stopPropagation();  // keys do not drive the robot
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); onRun(); }
    };
    this.box.oninput = () => onChange(this.value);
    parent.append(this.box);
    this.view = null;
    this.ready = import('../vendor/codemirror/codemirror.mjs').then((cm) => this.upgrade(cm))
      .catch((err) => console.warn('code editor not loaded, a plain text box instead:', err));
  }

  get value() {
    return this.view ? this.view.state.doc.toString() : this.box.value;
  }

  set value(text) {
    if (!this.view) { this.box.value = text; return; }
    this.view.dispatch({ changes: { from: 0, to: this.view.state.doc.length, insert: text } });
  }

  // Checks the text again (after something it depends on changed: the
  // format, Python loaded, a run's error).
  relint() {
    if (this.view) this.view.dispatch(this.cm.setDiagnostics(this.view.state, this.diagnostics(this.view.state)));
  }

  // lint()'s errors as CodeMirror diagnostics of `state`.
  diagnostics(state) {
    const doc = state.doc;
    return this.lint(doc.toString()).filter((d) => d.line >= 1 && d.line <= doc.lines).map((d) => {
      const line = doc.line(d.line);
      const from = d.column ? Math.min(line.to, line.from + d.column - 1) : line.from;
      return { from, to: d.column ? Math.min(line.to, from + 1) || line.to : line.to, severity: 'error', message: d.message };
    });
  }

  // Highlights (and indents) as `format` from now on.
  setFormat(format) {
    this.format = format;
    if (this.view) this.view.dispatch({ effects: this.language.reconfigure(this.languages[format]) });
  }

  focus() {
    (this.view ?? this.box).focus();
  }

  upgrade(cm) {
    this.cm = cm;
    const { EditorView, EditorState, Compartment, Prec, keymap, basicSetup, indentWithTab, indentUnit, StreamLanguage,
            linter, lintGutter, python, xml, oneDark } = cm;
    const controls = new Set(TYPES.filter((t) => !LEAVES.includes(t)).map((t) => t.toLowerCase()));
    const text = StreamLanguage.define({
      name: 'behavior-tree',
      startState: () => ({ first: true }),
      token(stream, state) {
        if (stream.sol()) state.first = true;
        if (stream.eatSpace()) return null;
        if (stream.peek() === '#') { stream.skipToEnd(); return 'comment'; }
        if (state.first) {  // the node
          state.first = false;
          const w = stream.match(/^\S+/)[0].toLowerCase();
          return controls.has(w) ? 'keyword' : 'typeName';
        }
        if (stream.match(/^\w+(?==)/)) return 'propertyName';
        if (stream.eat('=')) return 'operator';
        if (stream.match(/^-?\d+(\.\d+)?(e-?\d+)?(deg)?(?=[\s,#]|$)/)) return 'number';
        if (stream.match(/^"[^"]*"|^'[^']*'/)) return 'string';
        if (!stream.match(/^[^\s=#]+/)) stream.next();
        return 'variableName';
      },
      languageData: {
        commentTokens: { line: '#' },
        autocomplete: TYPES.map((label) => ({ label, type: controls.has(label.toLowerCase()) ? 'keyword' : 'function' })),
      },
    });
    const languages = this.languages = {
      text: [text, indentUnit.of('  ')],
      xml: [xml(), indentUnit.of('  ')],
      python: [python(), indentUnit.of('    ')],
    };
    const language = this.language = new Compartment();
    const check = linter((view) => this.diagnostics(view.state), { delay: 400 });
    this.view = new EditorView({
      parent: this.box.parentElement,
      state: EditorState.create({
        doc: this.box.value,
        extensions: [
          Prec.highest(keymap.of([{ key: 'Mod-Enter', run: () => { this.onRun(); return true; } }, indentWithTab])),
          basicSetup,
          oneDark,
          language.of(languages[this.format]),
          check,
          lintGutter(),
          EditorView.theme({
            '&': { height: '260px', fontSize: '11px', border: '1px solid #555', borderRadius: '4px', backgroundColor: '#1e2127' },
            '.cm-scroller': { fontFamily: 'ui-monospace, Menlo, Consolas, monospace', lineHeight: '1.45' },
            '.cm-gutters': { backgroundColor: '#1e2127' },
          }, { dark: true }),
          EditorView.updateListener.of((u) => {
            if (u.docChanged) this.onChange(u.state.doc.toString());
          }),
        ],
      }),
    });
    this.view.dom.addEventListener('keydown', (e) => e.stopPropagation());  // keys do not drive the robot
    this.box.replaceWith(this.view.dom);
  }
}
