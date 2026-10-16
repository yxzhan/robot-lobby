// Trained policies a robot can run as its "Policy" controller (Teleop
// panel). Add a model by adding an entry to POLICIES:
//
//   id          stable key, saved in the robot's controller choice
//   name        shown in the panel's Policy select
//   about       one line on what it drives
//   defaultFor  regex on the robot file path (under mjcf_assets/robots/):
//               robots that start with this policy on
//   load()      fetches the weights once, shared by every robot using it
//   create(weights, prefix)  a controller for the robot whose names start
//               with `prefix`
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
    about: 'Lower-body walking (legs and waist, 50 Hz) from public/groot; the arms keep their targets (IK chains or the Stage sliders).',
    defaultFor: G1_MODELS,
    load: async () => loadGrootPolicy(await onnxRuntime(), (name) => `groot/${name}`),
    create: (weights, prefix) => new GrootController(weights, prefix),
  },
];

export const policyById = (id) => POLICIES.find((p) => p.id === id) ?? null;

// The policy a robot file starts with ('' for none).
export const defaultPolicy = (path) => POLICIES.find((p) => p.defaultFor?.test(path))?.id ?? '';

// Weights of policy `p`, loaded once; a failed load is retried next time.
const loaded = new Map();
export function loadPolicy(p, progress = () => {}) {
  if (!loaded.has(p.id)) {
    progress(`loading ${p.name}…`);
    loaded.set(p.id, p.load().catch((err) => { loaded.delete(p.id); throw err; }));
  }
  return loaded.get(p.id);
}
