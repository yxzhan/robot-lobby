// What the Behavior panel's editor uses of CodeMirror 6, bundled by
// scripts/vendor.mjs (esbuild) into public/vendor/codemirror/codemirror.mjs:
// CodeMirror is many small ES modules, which a static site without a
// bundler cannot import by name.
export { basicSetup } from 'codemirror';
export { EditorView, keymap } from '@codemirror/view';
export { Compartment, EditorState, Prec } from '@codemirror/state';
export { indentWithTab } from '@codemirror/commands';
export { StreamLanguage, indentUnit } from '@codemirror/language';
export { lintGutter, linter, setDiagnostics } from '@codemirror/lint';
export { python } from '@codemirror/lang-python';
export { xml } from '@codemirror/lang-xml';
export { oneDark } from '@codemirror/theme-one-dark';
