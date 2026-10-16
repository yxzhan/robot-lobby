// Trained policies a robot can run as its "Policy" controller (Teleop
// panel). Their files live in pretrained_policies/<dir>/ (one checkout per
// model), which scripts/vendor.mjs links as public/pretrained_policies and
// lists in public/policies.json. Add a model by putting its directory there
// and adding an entry to POLICIES:
//
//   id          stable key, saved in the robot's controller choice
//   name        shown in the panel's Policy select
//   about       one line on what it drives
//   dir         its directory under pretrained_policies/
//   defaultFor  regex on the robot file path (under mjcf_assets/robots/):
//               robots that start with this policy on (if it is installed)
//   load(file)  fetches the weights once, shared by every robot using it;
//               file(name) is the URL of one of the files in `dir`
//   create(weights, prefix)  a controller for the robot whose names start
//               with `prefix`
//   physics     optional World settings it needs (physics.js keys, e.g.
//               {cone, impratio}); the Policy section offers to apply them
//
// A policy controller implements, like GrootController (groot.js):
//   name
//   bind(mujoco, model)  find its joints/actuators in a freshly compiled
//                        world; throw if the robot lacks them
//   unbind()             give back whatever bind changed (actuator gains)
//   owns(j)              its name if it drives joint j (IK leaves those alone)
//   pose(data)           the start pose at spawn
//   reset()              forget its history (robot reset)
//   async step(data, valid)  one tick every CONTROL_DT; write data.ctrl only
//                            if valid() still holds after any await
// Locomotion policies also take `cmd` = [vx, vy, yaw rate] with key speeds
// `drive` ({forward, back, side, turn}) and may have a commanded `height`
// (R/F keys); a hand or arm policy leaves those out and the keyboard skips it.

import { G1_MODELS, GrootController, loadGrootPolicy } from './groot.js';

// onnxruntime-web, loaded on first use.
let ort = null;
export function onnxRuntime() {
  ort ??= (async () => {
    const m = await import('./vendor/onnxruntime/ort.wasm.min.mjs');
    m.env.wasm.wasmPaths = new URL('vendor/onnxruntime/', location.href).href;
    m.env.wasm.numThreads = 1;  // tiny MLPs; threads would need cross-origin isolation
    return m;
  })();
  return ort;
}

export const POLICIES = [
  {
    id: 'groot-g1',
    name: 'GR00T locomotion · Unitree G1',
    about: 'Lower-body walking (legs and waist, 50 Hz); the arms keep their targets (IK chains or the Stage sliders).',
    dir: 'GR00T-WholeBodyControl_g1',  // huggingface.co/nepyope/GR00T-WholeBodyControl_g1
    defaultFor: G1_MODELS,
    load: async (file) => loadGrootPolicy(await onnxRuntime(), file),
    create: (weights, prefix) => new GrootController(weights, prefix),
  },
];

export const POLICY_ROOT = 'pretrained_policies';

export const policyById = (id) => POLICIES.find((p) => p.id === id) ?? null;

// The installed policy directories ({dir: [files]}), from public/policies.json.
let installed = {};
export async function loadPolicyIndex(index = 'policies.json') {
  try {
    const res = await fetch(index);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    installed = await res.json();
  } catch (err) {
    console.warn(`no policy index (${index}): run node scripts/vendor.mjs`, err);
    installed = {};
  }
}
export const isInstalled = (p) => !!p && p.dir in installed;

// The installed policy a robot file starts with ('' for none).
export const defaultPolicy = (path) => POLICIES.find((p) => p.defaultFor?.test(path) && isInstalled(p))?.id ?? '';

// Weights of policy `p`, loaded once; a failed load is retried next time.
const loaded = new Map();
export function loadPolicy(p, progress = () => {}) {
  if (!isInstalled(p)) return Promise.reject(new Error(`${POLICY_ROOT}/${p.dir}/ is not installed`));
  if (!loaded.has(p.id)) {
    progress(`loading ${p.name}…`);
    const file = (name) => `${POLICY_ROOT}/${p.dir}/${name}`;
    loaded.set(p.id, p.load(file).catch((err) => { loaded.delete(p.id); throw err; }));
  }
  return loaded.get(p.id);
}
