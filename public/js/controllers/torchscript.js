// Reads the tensors of a TorchScript archive (torch.jit.save, a .pt file) in
// the browser, without torch: the archive is a zip whose <name>/data/<i>
// entries are the raw little-endian storages, stored uncompressed, numbered
// in the order of the module's state_dict (parameters and buffers, as
// declared). Enough for small exported MLPs and RNNs whose layout the
// caller knows; it checks each storage's size against the expected shape.

// The storages of the archive at `url`, as Float32Arrays by index.
export async function readTorchScript(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return storages(await res.arrayBuffer(), url);
}

function storages(buf, what) {
  const v = new DataView(buf), u8 = new Uint8Array(buf);
  // End of central directory: the last PK\5\6 record (a zip64 archive
  // would need more; TorchScript files of this size never are).
  let eocd = -1;
  for (let i = buf.byteLength - 22; i >= Math.max(0, buf.byteLength - 65557); i--) {
    if (v.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error(`${what}: not a zip archive (TorchScript .pt)`);
  const count = v.getUint16(eocd + 10, true);
  let p = v.getUint32(eocd + 16, true);
  const out = [];
  const text = new TextDecoder();
  for (let n = 0; n < count; n++) {
    if (v.getUint32(p, true) !== 0x02014b50) throw new Error(`${what}: bad zip directory`);
    const method = v.getUint16(p + 10, true);
    const size = v.getUint32(p + 20, true);
    const nameLen = v.getUint16(p + 28, true), extraLen = v.getUint16(p + 30, true), commentLen = v.getUint16(p + 32, true);
    const local = v.getUint32(p + 42, true);
    const name = text.decode(u8.subarray(p + 46, p + 46 + nameLen));
    p += 46 + nameLen + extraLen + commentLen;
    const m = /\/data\/(\d+)$/.exec(name);
    if (!m) continue;
    if (method !== 0) throw new Error(`${what}: ${name} is compressed`);
    const start = local + 30 + v.getUint16(local + 26, true) + v.getUint16(local + 28, true);
    // Copy: the storages are 64-byte aligned in practice, but need not be.
    out[Number(m[1])] = new Float32Array(buf.slice(start, start + size));
  }
  return out;
}

// Checks `tensors` against `shapes` ([[name, ...dims]], in storage order)
// and returns them by name.
export function named(tensors, shapes, what) {
  if (tensors.length !== shapes.length) throw new Error(`${what}: ${tensors.length} tensors, expected ${shapes.length}`);
  const out = {};
  shapes.forEach(([name, ...dims], i) => {
    const n = dims.reduce((a, b) => a * b, 1);
    if (tensors[i]?.length !== n) throw new Error(`${what}: ${name} has ${tensors[i]?.length} values, expected ${dims.join('x')}`);
    out[name] = tensors[i];
  });
  return out;
}
