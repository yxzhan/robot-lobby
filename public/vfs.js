// Getting multi-file MJCF models (menagerie robots) into
// MuJoCo's Emscripten filesystem so that MjModel.from_xml_path() can resolve
// <include>, meshdir/texturedir and asset files exactly like it does natively.
//
// Image textures are the exception: MuJoCo would decode every PNG into its
// WASM heap (capped at 2 GB) although physics never reads them, so 2D file
// textures are swapped for 1x1 placeholders of the same name and the renderer
// decodes the images in the browser instead (see `images` below).

// Joins and normalizes '/'-separated paths ('.', '..', duplicate slashes).
export function joinPath(...parts) {
  const out = [];
  for (const seg of parts.join('/').split('/')) {
    if (seg === '..') out.pop();
    else if (seg && seg !== '.') out.push(seg);
  }
  return (parts[0].startsWith('/') ? '/' : '') + out.join('/');
}

function writeFile(mujoco, path, bytes) {
  mujoco.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
  mujoco.FS.writeFile(path, bytes);
}

function exists(mujoco, path) {
  return mujoco.FS.analyzePath(path).exists;
}

// A source maps a relative path to its bytes (rejecting if it does not exist).
export function urlSource(baseUrl) {
  return async (rel) => {
    const res = await fetch(`${baseUrl}/${rel}`);
    if (!res.ok) throw new Error(`${rel}: HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  };
}

// Reads the first candidate path that exists.
export async function readFirst(read, cands) {
  for (const [i, c] of cands.entries()) {
    try { return await read(c); } catch (err) { if (i === cands.length - 1) throw err; }
  }
}

// Attributes that name files, by element. `dir` says which compiler directory
// they are relative to (MuJoCo resolves hfield/skin/flexcomp via meshdir).
const FILE_ATTRS = [
  ['mesh', ['file'], 'meshdir'],
  ['skin', ['file'], 'meshdir'],
  ['hfield', ['file'], 'meshdir'],
  ['flexcomp', ['file'], 'meshdir'],
  ['texture', ['file', 'fileright', 'fileleft', 'fileup', 'filedown', 'filefront', 'fileback'], 'texturedir'],
];

// A plain 2D image texture the renderer can load itself. Anything with
// flips, grids or cube faces stays with MuJoCo.
function isPlainImageTexture(el) {
  return el.getAttribute('type') === '2d' && el.hasAttribute('file') &&
    !['gridsize', 'gridlayout', 'hflip', 'vflip'].some((a) => el.hasAttribute(a));
}

// Copies `entry` (a path in `read`'s namespace) and every file it references
// into the MuJoCo FS under `fsRoot`. Asset files already in the FS are not
// read again; XML files always are (they are rewritten, see above).
// Returns {path, images, docs}: the FS path of the entry file, a Map from
// texture name to candidate image paths for the textures replaced by
// placeholders, and the parsed XML documents of the include tree.
export async function loadModelFiles(mujoco, read, fsRoot, entry, onProgress = () => {}) {
  const modelDir = entry.includes('/') ? entry.slice(0, entry.lastIndexOf('/')) : '.';
  const images = new Map();

  // 1. The XML include tree. Include paths are relative to the including file.
  const docs = [];
  const seen = new Set();
  let pending = [entry];
  while (pending.length) {
    const batch = pending.filter((f) => !seen.has(f));
    batch.forEach((f) => seen.add(f));
    pending = [];
    onProgress(`fetching ${batch.join(', ')}`);
    const texts = await Promise.all(batch.map(async (f) => new TextDecoder().decode(await read(f))));
    for (const [i, text] of texts.entries()) {
      const fileDir = batch[i].slice(0, batch[i].lastIndexOf('/') + 1) || '.';
      const doc = new DOMParser().parseFromString(text, 'application/xml');
      const err = doc.querySelector('parsererror');
      if (err) throw new Error(`XML parse error: ${err.textContent}`);
      docs.push({ doc, fileDir, rel: batch[i] });
      for (const inc of doc.getElementsByTagName('include')) {
        pending.push(joinPath(fileDir, inc.getAttribute('file')));
      }
    }
  }

  // 2. Compiler directories (the <compiler> may live in any included file).
  const dirs = { assetdir: '', meshdir: '', texturedir: '' };
  for (const { doc } of docs) {
    for (const c of doc.getElementsByTagName('compiler')) {
      for (const k of Object.keys(dirs)) if (c.hasAttribute(k)) dirs[k] = c.getAttribute(k);
    }
  }
  dirs.meshdir ||= dirs.assetdir;
  dirs.texturedir ||= dirs.assetdir;

  // 3. Asset files: relative to the model dir + compiler dir, but MuJoCo also
  // finds them next to the included file that names them, so try both.
  const files = new Map();  // first candidate -> all candidates
  for (const { doc, fileDir } of docs) {
    for (const [tag, attrs, dir] of FILE_ATTRS) {
      for (const el of doc.getElementsByTagName(tag)) {
        const external = tag === 'texture' && isPlainImageTexture(el);
        for (const a of attrs) {
          const f = el.getAttribute(a);
          if (!f) continue;
          const cands = [...new Set([joinPath(modelDir, dirs[dir], f), joinPath(fileDir, dirs[dir], f)])];
          if (!external) { files.set(cands[0], cands); continue; }
          // An unnamed texture is named after its file stem; keep that name.
          const name = el.getAttribute('name') || f.split('/').pop().replace(/\.[^.]*$/, '');
          images.set(name, cands);
          el.setAttribute('name', name);
          el.removeAttribute('file');
          el.removeAttribute('content_type');
          el.setAttribute('builtin', 'flat');
          el.setAttribute('width', '1');
          el.setAttribute('height', '1');
          el.setAttribute('rgb1', '1 1 1');
        }
      }
    }
  }
  for (const { doc, rel } of docs) {
    writeFile(mujoco, joinPath(fsRoot, rel), new TextEncoder().encode(new XMLSerializer().serializeToString(doc)));
  }

  const todo = [...files.values()].filter((cands) => !cands.some((c) => exists(mujoco, joinPath(fsRoot, c))));
  let done = 0;
  const total = todo.length;
  const worker = async () => {
    while (todo.length) {
      const cands = todo.pop();
      for (const [i, c] of cands.entries()) {
        try {
          writeFile(mujoco, joinPath(fsRoot, c), await read(c));
          break;
        } catch (err) {
          if (i === cands.length - 1) throw err;
        }
      }
      onProgress(`fetching assets ${++done}/${total}`);
    }
  };
  await Promise.all(Array.from({ length: 8 }, worker));

  // 4. Sub-models (<asset><model file=…>) have their own compiler settings.
  for (const { doc } of docs) {
    for (const el of doc.getElementsByTagName('model')) {
      const f = el.getAttribute('file');
      if (!f) continue;
      const sub = await loadModelFiles(mujoco, read, fsRoot, joinPath(modelDir, f), onProgress);
      for (const [name, cands] of sub.images) images.set(name, cands);
    }
  }

  return { path: joinPath(fsRoot, entry), images, docs: docs.map((d) => d.doc) };
}
