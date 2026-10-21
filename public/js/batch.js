// Batched policy inference (onnxruntime-web) for the controllers in
// groot.js, playground.js and microduck.js: the robots running the same ONNX
// session share one session.run per control tick. main.js starts every
// controller's step() in the same synchronous loop, and each step() asks for
// its action before its first await, so the requests made until the
// microtasks run are one batch. A model whose input has a fixed batch size
// of 1 (the Go2 and Spot exports before export_onnx.py made it dynamic, the
// Microduck one) runs them one after another instead.
//
// runBatched resolves to this observation's action (Float32Array).
const batches = new Map();  // session -> [{obs, resolve, reject}]
export function runBatched(ort, session, obs, input = session.inputNames[0], output = session.outputNames[0]) {
  return new Promise((resolve, reject) => {
    if (!batches.has(session)) {
      batches.set(session, []);
      queueMicrotask(() => flush(ort, session, input, output));
    }
    batches.get(session).push({ obs, resolve, reject });
  });
}

async function flush(ort, session, input, output) {
  const reqs = batches.get(session);
  batches.delete(session);
  const n = reqs[0].obs.length;
  const run = async (x, b) => (await session.run({ [input]: new ort.Tensor('float32', x, [b, n]) }))[output].data;
  const dim = session.inputMetadata?.[session.inputNames.indexOf(input)]?.shape?.[0];
  try {
    if (reqs.length === 1 || dim === 1) {
      for (const r of reqs) r.resolve(Float32Array.from(await run(r.obs, 1)));
      return;
    }
    const x = new Float32Array(reqs.length * n);
    reqs.forEach((r, i) => x.set(r.obs, i * n));
    const y = await run(x, reqs.length), m = y.length / reqs.length;
    reqs.forEach((r, i) => r.resolve(y.slice(i * m, (i + 1) * m)));
  } catch (err) {
    for (const r of reqs) r.reject(err);
  }
}
