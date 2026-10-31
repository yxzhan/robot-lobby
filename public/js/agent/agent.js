// The Agent panel: a chat with a model that works the lobby with tools, as
// the panels do: it spawns and removes models, switches controllers, drives
// robots by their goal markers, runs skills and behaviour trees, changes
// the physics, looks at the world (its state, a picture) and the view.
// Everything runs in this page: the conversation, the tools, and the model
// requests, to one of
//   claude  the Claude API (the Anthropic TypeScript SDK, public/vendor/
//           anthropic, loaded on first use), with an API key the user
//           enters, kept in this browser;
//   ollama  an Ollama server (its /api/chat, with tools; by default
//           http://localhost:11434, which must allow this page's origin:
//           OLLAMA_ORIGINS).
// The conversation is kept as the Claude API takes it (content blocks);
// for Ollama it is turned into its messages on each request. It is kept in
// this browser (CHAT) and shown again after a reload, its screenshots left
// out (too big for localStorage). Tools that remove things (remove,
// clear_scene, reset_scene) ask first while "Ask before removing" is on.

import { POLICIES } from '../controllers/policy.js';
import { setView, viewOf } from '../world/stage.js';

const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

const PROVIDERS = [['claude', 'Claude'], ['ollama', 'Ollama']];
// The Claude models until the Models API has listed the key's (claudeModels).
const CLAUDE_MODELS = [['claude-opus-5-5', 'Claude Opus 5.5'], ['claude-sonnet-5-5', 'Claude Sonnet 5.5'], ['claude-haiku-5-5', 'Claude Haiku 5.5']]
  .map(([id, name]) => ({ id, name, efforts: ['low', 'medium', 'high'] }));
const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];
const SETTINGS = 'mujoco-filament/agent';      // localStorage: {provider, model, effort, ask, ollamaUrl, ollamaModel}
const KEY = 'mujoco-filament/anthropic-key';   // localStorage: the Claude API key
const CHAT = 'mujoco-filament/agent-chat';     // localStorage: {provider, messages}, the conversation
const NO_PICTURE = '(a picture, not kept after a reload)';
const OLLAMA_URL = 'http://localhost:11434';
const OLLAMA_CONTEXT = 32768;  // tokens (num_ctx): the tools' descriptions alone are some 5000
const CLAUDE_MAX_TOKENS = 16000;
const FALLBACK_BETA = 'server-side-fallback-2026-07-01';  // fallbacks: "default" (not for Haiku)
const MAX_STEPS = 40;     // requests per message at most (a runaway loop stops)
const SHOT_SIZE = 640;    // px, the screenshot tool's picture (square)
const RESULT_CHARS = 20000;  // a tool result longer than this is cut

const round = (v, n = 3) => (typeof v === 'number' ? Number(v.toFixed(n)) : v);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const SYSTEM = `You operate a robot sandbox in the user's browser ("robot lobby"): MuJoCo physics with robots and objects from a model library, driven by trained walking policies, mobile bases and IK. You act with the tools; the user sees the 3D view and this chat.

The world: metres and radians, z up, the floor at z = 0; yaw is the heading about z, counter-clockwise from +x. Everything spawned has a name r1, r2, ... (numbered from r1 again once the scene is empty) and a label (its model's name). A robot "drives" when a walking policy or a mobile base is on: only those move by goals.

How things move: robots walk or drive to goal poses (set_goal, move; a goal is let go once reached), or run skills of their policy (use_skill, e.g. the Microduck's kick). For anything coordinated or reactive (several robots, following, repeating, waiting for conditions), write a behaviour tree with run_behavior_tree: a py_trees script using the lobby module, described in its tool. Objects (a ball, furniture) move only when pushed. A walking robot is slow (about 0.2-0.5 m/s); allow time, and use wait to let it get there.

Spawning: find a model with search_assets, then spawn it; walking robots (Unitree G1, Spot, Go2, Microduck) get their walking policy switched on by themselves where one is installed. Leave about a metre between robots.

Work step by step and check what happened with get_world_state, or screenshot when the picture matters (a fall, where things are); set_view first to look at the right spot. Keep replies short; answer in the user's language.`;

// The tools: [definition, run(input, ctx)]. A run returns what the tool
// gives back (JSON-able, or {content} for image blocks) or throws.
function tools(agent) {
  const { app } = agent;
  const goals = () => window.goals, lobby = () => window.lobby;
  const robotOf = (name) => {
    const r = lobby().robots().find((x) => x.prefix.slice(0, -1) === name || x.label === name);
    if (!r) throw new Error(`no ${name} (spawned: ${lobby().robots().map((x) => x.prefix.slice(0, -1)).join(', ') || 'nothing'})`);
    return r;
  };
  const which = (robots) => (robots == null || robots === 'all' ? 'all' : [robots].flat().map(String));
  const policyState = (r) => {
    const c = r.use?.policy && r.policy;
    const menu = c?.skillMenu;
    return {
      policy: r.use?.policy || null,
      skills: menu ? Object.fromEntries(menu.manual.map((n) => [n, { label: menu.skills[n]?.label, ...(menu.skills[n]?.duration && { duration: menu.skills[n].duration }) }])) : null,
      active_skill: c?.activeSkill ?? null,
    };
  };
  const state = () => {
    const list = goals().list();
    const poses = new Map(list.map((g) => [g.name, g]));
    return {
      sim_time: round(app.data?.time ?? 0, 2),
      paused: !!app.paused,
      shared_world: !!app.net,
      physics: Object.fromEntries(['timestep', 'speed', 'gravity', 'noslip_iterations', 'integrator'].map((k) => [k, app.physics?.[k]])),
      things: lobby().robots().map((r) => {
        const name = r.prefix.slice(0, -1), g = poses.get(name);
        return {
          name, label: r.label,
          pose: g && { x: round(g.pose.x), y: round(g.pose.y), yaw: round(g.pose.yaw) },
          drives: !!g?.drives,
          goal: g?.goal && { x: round(g.goal.x), y: round(g.goal.y), yaw: round(g.goal.yaw) },
          arrived: !!g?.arrived,
          ...(r.use && { controllers: { base: !!r.use.base, ik: !!r.use.ik }, ...policyState(r) }),
        };
      }),
      behavior_tree: window.bt.status() ? { status: window.bt.status().status, error: window.bt.status().error } : null,
      view: (({ center, dist, azim, elev }) => ({ center: center.map((v) => round(v, 2)), dist: round(dist, 2), azim: round(azim, 2), elev: round(elev, 2) }))(viewOf(app.orbit)),
    };
  };
  const num = (v, what) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`${what}: a number`);
    return v;
  };
  const robotsProp = { description: "'all' (every robot that drives), a name (r1), a label, or a list of those", anyOf: [{ type: 'string' }, { type: 'array', items: { type: 'string' } }] };

  return [
    [{
      name: 'get_world_state',
      description: 'What is in the world now: every spawned thing (name, label, pose, whether it drives, its goal, its controllers, policy and skills), the simulation time, the physics, the behaviour tree running, the view.',
      input_schema: { type: 'object', properties: {} },
    }, () => state()],

    [{
      name: 'inspect',
      description: "What a spawned thing is made of: its bodies as a tree (name, parent, world pos and rpy, mass, joints with type, qpos (rad or m), qvel and range, size: the extent of its collision geoms in the body's own frame [x, y, z] m around center, and those geoms: type and MuJoCo size), its actuators (ctrl, range), its total mass and its bounds in the world (collision geoms, min / max). For heights to put things on, reach, what a joint is at. `bodies`: only bodies whose name has this text (a big robot lists many).",
      input_schema: { type: 'object', properties: { name: { type: 'string' }, bodies: { type: 'string' } }, required: ['name'] },
    }, ({ name, bodies }) => lobby().describe(robotOf(name), bodies || undefined)],

    [{
      name: 'search_assets',
      description: 'Searches the model library by words of a name (case-insensitive; empty: everything of the category). Returns up to `limit` models: {rel (what spawn takes), name, category (robot, object, environment), source, group}.',
      input_schema: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'e.g. "microduck", "g1", "ball", "table"' },
          category: { type: 'string', enum: ['robot', 'object', 'environment'] },
          limit: { type: 'integer', description: 'default 20' },
        },
        required: ['query'],
      },
    }, ({ query, category, limit = 20 }) => {
      const hits = lobby().search(query, category);
      return { count: hits.length, models: hits.slice(0, limit) };
    }],

    [{
      name: 'spawn',
      description: 'Spawns a library model (its rel from search_assets) at x, y, turned yaw: standing on the floor, or with z its base that high (m: on a table, a shelf). Returns the new thing as get_world_state lists it. Takes a few seconds (the world is rebuilt).',
      input_schema: {
        type: 'object',
        properties: { rel: { type: 'string' }, x: { type: 'number' }, y: { type: 'number' }, yaw: { type: 'number', description: 'rad, default 0' },
          z: { type: 'number', description: 'm, default 0 (the floor)' } },
        required: ['rel', 'x', 'y'],
      },
    }, async ({ rel, x, y, yaw = 0, z = 0 }) => {
      const r = await lobby().spawn(rel, [num(x, 'x'), num(y, 'y'), num(z, 'z')], num(yaw, 'yaw'));
      await sleep(300);
      return state().things.find((t) => t.name === r.prefix.slice(0, -1)) ?? { spawned: r.prefix.slice(0, -1) };
    }],

    [{
      name: 'remove',
      description: 'Removes spawned things by name or label (the others stay as they are).',
      input_schema: { type: 'object', properties: { names: { type: 'array', items: { type: 'string' } } }, required: ['names'] },
    }, async ({ names }) => {
      const list = names.map(robotOf);
      if (!agent.confirm(`The agent wants to remove ${list.map((r) => r.prefix.slice(0, -1)).join(', ')}.`)) throw new Error('the user declined');
      await lobby().remove(list);
      return { removed: list.map((r) => r.prefix.slice(0, -1)) };
    }],

    [{
      name: 'clear_scene',
      description: 'Removes everything spawned (names start again at r1).',
      input_schema: { type: 'object', properties: {} },
    }, async () => {
      const n = lobby().robots().length;
      if (n && !agent.confirm(`The agent wants to remove all ${n} spawned models.`)) throw new Error('the user declined');
      await lobby().remove(lobby().robots());
      return { removed: n };
    }],

    [{
      name: 'reset_scene',
      description: 'Puts every spawned thing back where it was spawned, at rest, and the simulation time back to 0 (goals and controllers start over).',
      input_schema: { type: 'object', properties: {} },
    }, async () => {
      if (!agent.confirm('The agent wants to reset the scene.')) throw new Error('the user declined');
      lobby().reset();
      await sleep(500);
      return { reset: true, sim_time: round(app.data?.time ?? 0, 2) };
    }],

    [{
      name: 'set_paused',
      description: 'Pauses or resumes the simulation.',
      input_schema: { type: 'object', properties: { paused: { type: 'boolean' } }, required: ['paused'] },
    }, async ({ paused }) => { await lobby().setPaused(paused); return { paused: !!app.paused }; }],

    [{
      name: 'set_controller',
      description: `Switches a robot's controller on or off: "policy" (a trained walking policy; on: its id, by default the one for this robot), "base" (a mobile base), "ik" (IK chains for arms). Policies: ${POLICIES.map((p) => `${p.id} (${p.name})`).join('; ')}.`,
      input_schema: {
        type: 'object',
        properties: {
          robot: { type: 'string' }, kind: { type: 'string', enum: ['policy', 'base', 'ik'] }, on: { type: 'boolean' },
          policy: { type: 'string', description: 'for kind "policy": the policy id' },
        },
        required: ['robot', 'kind', 'on'],
      },
    }, async ({ robot, kind, on, policy }) => {
      const r = robotOf(robot);
      await lobby().setController(r, kind, on, policy);
      await sleep(app.net ? 800 : 100);
      return state().things.find((t) => t.name === r.prefix.slice(0, -1));
    }],

    [{
      name: 'set_goal',
      description: 'Sends robots to a pose on the floor (world frame): they walk or drive there and stop; yaw is the heading to end with (default: the way there). Returns at once; use wait to let them get there.',
      input_schema: {
        type: 'object',
        properties: { robots: robotsProp, x: { type: 'number' }, y: { type: 'number' }, yaw: { type: 'number' } },
        required: ['robots', 'x', 'y'],
      },
    }, ({ robots, x, y, yaw }) => goalsTo(robots, num(x, 'x'), num(y, 'y'), yaw == null ? null : num(yaw, 'yaw'))],

    [{
      name: 'move',
      description: "Moves robots relative to where each one is: forward and left in metres in its own frame, turn in radians (counter-clockwise). Returns at once; use wait to let them get there.",
      input_schema: {
        type: 'object',
        properties: { robots: robotsProp, forward: { type: 'number' }, left: { type: 'number' }, turn: { type: 'number' } },
        required: ['robots'],
      },
    }, ({ robots, forward = 0, left = 0, turn = 0 }) => goals().move(which(robots), { forward, left, turn })
      .map((g) => ({ name: g.name, goal: g.goal && { x: round(g.goal.x), y: round(g.goal.y), yaw: round(g.goal.yaw) } }))],

    [{
      name: 'stop',
      description: 'Clears robots\' goals: they stop where they are.',
      input_schema: { type: 'object', properties: { robots: robotsProp }, required: ['robots'] },
    }, ({ robots }) => goals().clear(which(robots)).map((g) => g.name)],

    [{
      name: 'wait',
      description: 'Lets the simulation run for `seconds` of simulated time (at most 60), or until the robots in `until_arrived` have reached their goals, whichever is first. Returns where those robots (or all) are.',
      input_schema: {
        type: 'object',
        properties: { seconds: { type: 'number' }, until_arrived: robotsProp },
        required: ['seconds'],
      },
    }, async ({ seconds, until_arrived }, signal) => {
      const s = Math.min(60, Math.max(0, num(seconds, 'seconds')));
      const t0 = app.data?.time ?? 0, w0 = performance.now();
      const names = until_arrived == null ? null : new Set(goals().list().filter((g) => {
        const w = which(until_arrived);
        return w === 'all' ? g.drives : w.includes(g.name) || w.includes(g.label);
      }).map((g) => g.name));
      let why = 'time';
      for (;;) {
        if (signal.aborted) throw new Error('stopped');
        const t = app.data?.time ?? 0;
        if (t - t0 >= s || t < t0) break;
        if (names && goals().list().filter((g) => names.has(g.name)).every((g) => !g.goal)) { why = 'arrived'; break; }
        if (performance.now() - w0 > 1000 * (2 * s / Math.max(0.1, app.physics?.speed || 1) + 30)) { why = app.paused ? 'paused' : 'slow'; break; }
        await sleep(100);
      }
      const list = goals().list().filter((g) => !names || names.has(g.name));
      return {
        waited: round((app.data?.time ?? 0) - t0, 2), stopped_by: why,
        robots: list.map((g) => ({ name: g.name, pose: { x: round(g.pose.x), y: round(g.pose.y), yaw: round(g.pose.yaw) }, goal: g.goal && { x: round(g.goal.x), y: round(g.goal.y) }, arrived: g.arrived })),
      };
    }],

    [{
      name: 'use_skill',
      description: "Runs one of a robot's policy skills (get_world_state lists them per robot, e.g. kick_right for the Microduck): a timed one for its duration (or `duration` s), then back to the default (walking).",
      input_schema: {
        type: 'object',
        properties: { robot: { type: 'string' }, skill: { type: 'string' }, duration: { type: 'number' } },
        required: ['robot', 'skill'],
      },
    }, async ({ robot, skill, duration }) => {
      const r = robotOf(robot), c = r.use?.policy && r.policy;
      if (!c?.skillMenu) throw new Error(`${robot} has no skills to pick (policy: ${r.use?.policy || 'off'})`);
      c.setSkill(skill, duration ?? agent.app.teleop?.skillDurations?.[r.use.policy]?.[skill]);
      await sleep(200);
      return policyState(r);
    }],

    [{
      name: 'run_behavior_tree',
      description: `Runs a behaviour tree in the Behavior panel (replacing one that runs): a py_trees 2.6 script (Python, in the page) that defines \`root\` (a behaviour) or \`tree\` (a BehaviourTree). Ticked every 0.1 s of simulated time while the simulation runs; check it with behavior_tree_status. The lobby module:\n\n${agent.lobbyDoc}`,
      input_schema: { type: 'object', properties: { script: { type: 'string' } }, required: ['script'] },
    }, async ({ script }) => {
      await window.bt.run(script, 'python', { confirm: (what) => agent.confirm(what) });  // (its removing and resetting ask too)
      await sleep(600);
      return window.bt.status() ?? { status: 'not started' };
    }],

    [{
      name: 'behavior_tree_status',
      description: "The behaviour tree's status: running, success, failure (with the error and its line), and each node's status and feedback.",
      input_schema: { type: 'object', properties: {} },
    }, () => window.bt.status() ?? { status: 'none' }],

    [{
      name: 'stop_behavior_tree',
      description: 'Stops the behaviour tree; the robots it drives stop where they are.',
      input_schema: { type: 'object', properties: {} },
    }, () => { window.bt.stop(); return window.bt.status() ?? { status: 'none' }; }],

    [{
      name: 'set_physics',
      description: 'Changes physics options (the Physics panel): gravity [x, y, z] (m/s²), timestep (s, 0.0001-0.05), speed (simulated per wall second; 0: as fast as it goes), noslip_iterations, integrator (Euler, RK4, implicit, implicitfast), density and viscosity of the medium, wind [x, y, z]. Returns the physics now.',
      input_schema: { type: 'object', properties: { changes: { type: 'object' } }, required: ['changes'] },
    }, async ({ changes }) => {
      await lobby().setPhysics(changes);
      return Object.fromEntries(Object.keys(changes).map((k) => [k, app.physics[k]]));
    }],

    [{
      name: 'set_view',
      description: "Points the user's view (the camera orbiting a point): center [x, y, z] it looks at, dist (m), azim (rad, about z), elev (rad above the floor, up to 1.5). Leave out what should stay.",
      input_schema: {
        type: 'object',
        properties: { center: { type: 'array', items: { type: 'number' } }, dist: { type: 'number' }, azim: { type: 'number' }, elev: { type: 'number' } },
      },
    }, async (v) => {
      const now = viewOf(app.orbit);
      setView(app.orbit, { ...now, ...v, center: v.center ?? now.center });
      await sleep(200);
      return state().view;
    }],

    [{
      name: 'screenshot',
      description: 'A picture of what the user sees in the 3D view now (square, the main camera).',
      input_schema: { type: 'object', properties: {} },
    }, async () => {
      const blob = await lobby().capture(SHOT_SIZE);
      if (!blob) throw new Error('no picture (the view did not render)');
      const data = await new Promise((resolve) => {
        const r = new FileReader();
        r.onload = () => resolve(String(r.result).split(',')[1]);
        r.readAsDataURL(blob);
      });
      return { content: [{ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data } }, { type: 'text', text: `the view at sim time ${round(app.data?.time ?? 0, 2)} s` }], image: data };
    }],
  ];

  // set_goal: yaw given or each robot's way there.
  function goalsTo(robots, x, y, yaw) {
    const w = which(robots), list = goals().list().filter((g) => (w === 'all' ? g.drives : w.includes(g.name) || w.includes(g.label)));
    if (!list.length) throw new Error(`no robot ${[w].flat().join(', ')} that drives`);
    return list.map((g) => {
      goals().set(g.name, { x, y, yaw: yaw ?? Math.atan2(y - g.pose.y, x - g.pose.x) });
      return { name: g.name, goal: { x, y, yaw: round(yaw ?? Math.atan2(y - g.pose.y, x - g.pose.x)) } };
    });
  }
}

// The Anthropic SDK (bundled; loaded on first use).
let sdk = null;
const anthropicSdk = () => (sdk ??= import('../../vendor/anthropic/anthropic.mjs').then((m) => m.default));

// The Claude models this key can use (the Models API, newest first) that
// think adaptively (as claudeTurn asks), each with the efforts it takes.
async function claudeModels(key) {
  const Anthropic = await anthropicSdk();
  const client = new Anthropic({ apiKey: key, dangerouslyAllowBrowser: true });
  const out = [];
  for await (const m of client.models.list({ limit: 100 })) {
    const caps = m.capabilities;
    if (caps && !caps.thinking?.types?.adaptive?.supported) continue;
    const efforts = caps?.effort ? EFFORTS.filter((e) => caps.effort[e]?.supported) : [];
    out.push({ id: m.id, name: m.display_name || m.id, efforts: efforts.length ? efforts : ['low', 'medium', 'high'] });
  }
  return out;
}

// A Claude API request: the reply message (content blocks, stop_reason, usage).
async function claudeTurn({ key, model, effort, system, tools: defs, messages, signal }) {
  const Anthropic = await anthropicSdk();
  const client = new Anthropic({ apiKey: key, dangerouslyAllowBrowser: true });  // (the user's own key, in their browser)
  const params = {
    model, max_tokens: CLAUDE_MAX_TOKENS, system, tools: defs, messages,
    thinking: { type: 'adaptive', display: 'summarized' },
    output_config: { effort },
    cache_control: { type: 'ephemeral' },  // the system prompt, the tools and the history so far
  };
  if (!model.startsWith('claude-haiku')) Object.assign(params, { betas: [FALLBACK_BETA], fallbacks: 'default' });
  try {
    return await client.beta.messages.create(params, { signal });
  } catch (err) {
    if (err instanceof Anthropic.AuthenticationError) throw new Error('the API key was not accepted (401): check it in the panel');
    if (err instanceof Anthropic.PermissionDeniedError) throw new Error(`not allowed (403): ${err.message}`);
    if (err instanceof Anthropic.RateLimitError) throw new Error('rate limited (429): try again in a moment');
    if (err instanceof Anthropic.APIError && err.status) throw new Error(`${err.status}: ${err.error?.error?.message ?? err.message}`);
    throw err;
  }
}

// The conversation (Claude API form) as Ollama's messages; screenshots as
// images of a user message when the model sees (`vision`).
function ollamaMessages(system, messages, vision) {
  const out = [{ role: 'system', content: system }], names = new Map();  // tool_use id -> tool name
  for (const m of messages) {
    if (typeof m.content === 'string') { out.push({ role: m.role, content: m.content }); continue; }
    if (m.role === 'assistant') {
      const calls = m.content.filter((b) => b.type === 'tool_use');
      for (const c of calls) names.set(c.id, c.name);
      out.push({
        role: 'assistant',
        content: m.content.filter((b) => b.type === 'text').map((b) => b.text).join('\n'),
        ...(calls.length && { tool_calls: calls.map((c) => ({ function: { name: c.name, arguments: c.input } })) }),
      });
      continue;
    }
    for (const b of m.content) {
      if (b.type === 'text') out.push({ role: 'user', content: b.text });
      if (b.type !== 'tool_result') continue;
      const parts = typeof b.content === 'string' ? [{ type: 'text', text: b.content }] : b.content;
      const text = parts.filter((c) => c.type === 'text').map((c) => c.text).join('\n');
      const images = parts.filter((c) => c.type === 'image').map((c) => c.source.data);
      const unseen = images.length && !vision ? ' (a picture this model cannot see)' : '';
      out.push({ role: 'tool', tool_name: names.get(b.tool_use_id), content: b.is_error ? `error: ${text}` : text + unseen });
      if (images.length && vision) out.push({ role: 'user', content: 'The screenshot:', images });
    }
  }
  return out;
}

let ollamaIds = 0;

// An Ollama request, its reply turned into a Claude API message.
async function ollamaTurn({ url, model, info, system, tools: defs, messages, signal }) {
  const res = await fetch(`${url}/api/chat`, {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model, stream: false,
      messages: ollamaMessages(system, messages, info?.vision),
      tools: defs.map((d) => ({ type: 'function', function: { name: d.name, description: d.description, parameters: d.input_schema } })),
      options: { num_ctx: OLLAMA_CONTEXT },
      ...(info?.thinking && { think: true }),
    }),
  }).catch((err) => { throw signal.aborted ? err : new Error(`Ollama is not reachable at ${url} (${err.message}): is it running, and does OLLAMA_ORIGINS allow ${location.origin}?`); });
  const body = await res.json().catch(() => ({}));
  if (!res.ok || body.error) throw new Error(`Ollama: ${body.error ?? `${res.status} ${res.statusText}`}`);
  const m = body.message ?? {}, content = [];
  if (m.thinking?.trim()) content.push({ type: 'thinking', thinking: m.thinking });
  if (m.content?.trim()) content.push({ type: 'text', text: m.content });
  for (const c of m.tool_calls ?? []) {
    let input = c.function?.arguments ?? {};
    if (typeof input === 'string') { try { input = JSON.parse(input); } catch { input = {}; } }
    content.push({ type: 'tool_use', id: `ollama_${++ollamaIds}`, name: c.function?.name, input });
  }
  return {
    content,
    stop_reason: m.tool_calls?.length ? 'tool_use' : body.done_reason === 'length' ? 'max_tokens' : 'end_turn',
    usage: { input_tokens: body.prompt_eval_count ?? 0, output_tokens: body.eval_count ?? 0 },
  };
}

export class AgentPanel {
  constructor(app, root) {
    this.app = app;
    this.body = root.querySelector('.body');
    this.messages = [];      // the conversation, as the Claude API takes it
    this.running = null;     // {abort: AbortController} while a turn runs
    this.lobbyDoc = '';
    this.ollama = { models: [], info: null, error: null };  // what the Ollama server has
    this.claude = { models: CLAUDE_MODELS, error: null };   // what the key can use
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(SETTINGS)) ?? {}; } catch { /* none */ }
    this.settings = { provider: 'claude', model: CLAUDE_MODELS[0].id, effort: 'medium', ask: true, ollamaUrl: OLLAMA_URL, ollamaModel: '', ...saved };
    try { this.key = localStorage.getItem(KEY) ?? ''; } catch { this.key = ''; }

    const select = (options, value, onchange, title) => {
      const s = el('select', { title }, ...options.map(([v, label]) => new Option(label, v, false, v === value)));
      s.onchange = () => { onchange(s.value); s.blur(); };
      return s;
    };
    const set = (k) => (v) => {
      this.settings[k] = v;
      try { localStorage.setItem(SETTINGS, JSON.stringify(this.settings)); } catch { /* no storage */ }
      this.update();
    };
    const typing = (e) => e.stopPropagation();  // keys do not drive the robot

    // Provider and its settings.
    const provider = select(PROVIDERS, this.settings.provider, (v) => {
      if (this.messages.length) this.newChat('(a new chat: the provider changed)');  // their histories do not mix
      set('provider')(v);
      if (v === 'ollama') this.refreshOllama();
    }, 'Who answers: Claude (the Anthropic API, your key) or a model of an Ollama server');
    this.claudeModel = select([], this.settings.model, set('model'), 'The Claude model (those your API key can use)');
    this.effort = select([], this.settings.effort, set('effort'), 'How much Claude thinks before acting (more: slower, more careful)');
    const listModels = el('button', { type: 'button', textContent: '↻', title: 'Ask the Anthropic API again for the models' });
    listModels.onclick = () => this.refreshClaude();
    this.ollamaModel = el('select', { title: 'The Ollama model (one that supports tools)' });
    this.ollamaModel.onchange = () => { set('ollamaModel')(this.ollamaModel.value); this.ollamaModel.blur(); this.refreshOllama(); };
    const fresh = el('button', { type: 'button', textContent: 'New chat', title: 'Forget the conversation (the world stays as it is)' });
    fresh.onclick = () => this.newChat();

    this.keyInput = el('input', { type: 'password', placeholder: 'sk-ant-…', autocomplete: 'off', spellcheck: false,
      title: 'Your Claude API key (platform.claude.com → API keys): kept in this browser only, sent only to api.anthropic.com' });
    this.keyInput.onkeydown = typing;
    const saveKey = el('button', { type: 'submit', textContent: 'Save' });
    // A form of its own: without one the browser's password manager takes
    // the whole page for the login form, the Assets filter before it for
    // the user name, and offers to fill that in.
    const keyForm = el('form', { className: 'bar agent-key-entry', autocomplete: 'off' }, this.keyInput, saveKey);
    keyForm.onsubmit = (e) => {
      e.preventDefault();
      this.key = this.keyInput.value.trim();
      this.keyInput.value = '';
      try { localStorage.setItem(KEY, this.key); } catch { /* no storage: kept for this page only */ }
      this.update();
      this.refreshClaude();
    };
    const forget = el('button', { type: 'button', textContent: 'Forget key', title: 'Remove the API key from this browser' });
    forget.onclick = () => {
      this.key = '';
      this.claude = { models: CLAUDE_MODELS, error: null };
      try { localStorage.removeItem(KEY); } catch { /* none */ }
      this.update();
    };
    this.claudeNote = el('div', { className: 'hint' });
    this.claudeBox = el('div', { className: 'agent-provider' },
      el('div', { className: 'bar' }, this.claudeModel, this.effort, listModels),
      this.claudeNote,
      keyForm,
      el('div', { className: 'bar agent-key-saved' }, el('span', { className: 'note ok', textContent: 'API key saved in this browser' }), forget),
      el('div', { className: 'hint agent-key-help' }, 'A Claude API key from ', el('a', { href: 'https://platform.claude.com/settings/keys', target: '_blank', rel: 'noopener', textContent: 'platform.claude.com' }),
        ' (billed to that account): kept in this browser only and sent only to the Anthropic API. Set a spending limit there.'));

    this.urlInput = el('input', { type: 'text', value: this.settings.ollamaUrl, spellcheck: false, title: 'The Ollama server' });
    this.urlInput.onkeydown = (e) => { typing(e); if (e.key === 'Enter') this.urlInput.blur(); };
    this.urlInput.onchange = () => { set('ollamaUrl')(this.urlInput.value.trim().replace(/\/+$/, '') || OLLAMA_URL); this.refreshOllama(); };
    const refresh = el('button', { type: 'button', textContent: '↻', title: 'Ask the Ollama server again for its models' });
    refresh.onclick = () => this.refreshOllama();
    this.ollamaNote = el('div', { className: 'hint' });
    this.ollamaBox = el('div', { className: 'agent-provider' },
      el('div', { className: 'bar' }, this.urlInput, refresh),
      el('div', { className: 'bar' }, this.ollamaModel),
      this.ollamaNote);

    const ask = el('input', { type: 'checkbox', checked: this.settings.ask });
    ask.onchange = () => set('ask')(ask.checked);
    this.log = el('div', { className: 'agent-log' });
    this.input = el('textarea', { rows: 3, placeholder: 'Ask the agent: "spawn two microducks and have them swap places"…', spellcheck: true });
    this.input.onkeydown = (e) => {
      typing(e);
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); this.send(); }
    };
    this.sendButton = el('button', { type: 'button', className: 'primary', textContent: 'Send', title: 'Send (Enter; Shift+Enter: a new line)' });
    this.sendButton.onclick = () => (this.running ? this.stop() : this.send());
    this.note = el('span', { className: 'note' });
    this.body.append(
      el('div', { className: 'bar' }, provider, fresh),
      this.claudeBox, this.ollamaBox,
      this.log, this.input,
      el('div', { className: 'bar' }, this.sendButton,
        el('label', { className: 'agent-ask', title: 'Remove, clear and reset ask you first' }, ask, ' Ask before removing or resetting'),
        this.note));
    this.tools = tools(this);
    this.restore();
    this.update();
    if (this.settings.provider === 'ollama') this.refreshOllama();
    if (this.key) this.refreshClaude();
    // Opened again: the Ollama server may have models now.
    new MutationObserver(() => { if (!root.hidden && this.settings.provider === 'ollama' && !this.ollama.models.length) this.refreshOllama(); })
      .observe(root, { attributes: true, attributeFilter: ['hidden'] });
    fetch(new URL('../behavior/lobby.py', import.meta.url)).then((r) => r.text())
      .then((t) => { this.lobbyDoc = t.match(/^"""([\s\S]*?)"""/)?.[1].trim() ?? ''; this.tools = tools(this); })
      .catch(() => {});
  }

  // The Ollama server's models (/api/tags), and what the picked one can do
  // (/api/show: tools, vision, thinking).
  async refreshOllama() {
    const url = this.settings.ollamaUrl;
    const show = (model) => fetch(`${url}/api/show`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model }) })
      .then((r) => (r.ok ? r.json() : {}))
      .then((info) => {
        const caps = info.capabilities ?? [];
        return { tools: caps.includes('tools'), vision: caps.includes('vision'), thinking: caps.includes('thinking'), known: caps.length > 0 };
      });
    this.ollama = { models: this.ollama.models, info: null, error: null };
    try {
      const tags = await fetch(`${url}/api/tags`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))));
      this.ollama.models = (tags.models ?? []).map((m) => m.name).sort();
      let model = this.settings.ollamaModel, info = null;
      if (!this.ollama.models.includes(model)) {  // none picked yet (or gone): the first that has tools
        model = this.ollama.models[0] ?? '';
        for (const m of this.ollama.models.slice(0, 20)) {
          const i = await show(m);
          if (i.tools) { model = m; info = i; break; }
        }
        this.settings.ollamaModel = model;
        try { localStorage.setItem(SETTINGS, JSON.stringify(this.settings)); } catch { /* no storage */ }
      }
      if (model) this.ollama.info = info ?? await show(model);
    } catch (err) {
      this.ollama = { models: [], info: null, error: `not reachable at ${url} (${err.message ?? err})` };
    }
    this.update();
  }

  // The models the key can use (claudeModels); the built-in list if that fails.
  async refreshClaude() {
    if (!this.key) return;
    this.claude.error = null;
    try {
      const models = await claudeModels(this.key);
      if (models.length) this.claude.models = models;
    } catch (err) {
      this.claude.error = `could not list the models (${err.status ?? err.message ?? err}): the usual ones are offered`;
    }
    this.update();
  }

  // Whether a message can be sent now; else why not (for the panel).
  readiness() {
    if (this.settings.provider === 'claude') return this.key ? null : 'Enter a Claude API key to start.';
    const o = this.ollama;
    if (o.error) return `Ollama ${o.error}. Start it with OLLAMA_ORIGINS=${location.origin} ollama serve (or allow * ), then ↻.`;
    if (!this.settings.ollamaModel) return 'The Ollama server has no models: ollama pull a model with tools (e.g. qwen3), then ↻.';
    if (o.info?.known && !o.info.tools) return `${this.settings.ollamaModel} does not support tools: pick another model.`;
    return null;
  }

  confirm(what) {
    return !this.settings.ask || window.confirm(`${what} Allow it?`);
  }

  update() {
    const claude = this.settings.provider === 'claude';
    this.claudeBox.hidden = !claude;
    this.ollamaBox.hidden = claude;
    this.claudeBox.querySelector('.agent-key-entry').hidden = !!this.key;
    this.claudeBox.querySelector('.agent-key-help').hidden = !!this.key;
    this.claudeBox.querySelector('.agent-key-saved').hidden = !this.key;
    const models = this.claude.models;
    const model = models.find((m) => m.id === this.settings.model) ?? models[0];
    this.settings.model = model.id;  // (a model gone: the newest)
    if (!model.efforts.includes(this.settings.effort)) this.settings.effort = model.efforts.includes('medium') ? 'medium' : model.efforts[0];
    this.claudeModel.replaceChildren(...models.map((m) => new Option(m.name, m.id, false, m === model)));
    this.effort.replaceChildren(...model.efforts.map((e) => new Option(e, e, false, e === this.settings.effort)));
    this.claudeNote.textContent = this.key ? this.claude.error ?? '' : '';
    this.ollamaModel.replaceChildren(...this.ollama.models.map((m) => new Option(m, m, false, m === this.settings.ollamaModel)));
    this.ollamaModel.disabled = !this.ollama.models.length;
    const i = this.ollama.info;
    const why = this.readiness();
    this.ollamaNote.textContent = !claude && why ? why
      : i?.known ? `can: ${['tools', 'vision', 'thinking'].filter((k) => i[k]).join(', ') || 'none of tools, vision, thinking'}${i.vision ? '' : ' (screenshots are described, not shown)'}` : '';
    this.sendButton.textContent = this.running ? 'Stop' : 'Send';
    this.sendButton.className = this.running ? '' : 'primary';
    this.sendButton.disabled = !this.running && !!why;
  }

  stop() {
    this.running?.abort.abort();
  }

  newChat(note) {
    this.stop();
    this.messages = [];
    this.save();
    this.log.replaceChildren();
    if (note) this.entry('note', note);
    this.note.textContent = '';
  }

  // Keeps the conversation in this browser, its pictures left out.
  save() {
    const lean = (b) => (b.type === 'tool_result' && Array.isArray(b.content)
      ? { ...b, content: b.content.map((c) => (c.type === 'image' ? { type: 'text', text: NO_PICTURE } : c)) } : b);
    const messages = this.messages.map((m) => (Array.isArray(m.content) ? { ...m, content: m.content.map(lean) } : m));
    try {
      if (messages.length) localStorage.setItem(CHAT, JSON.stringify({ provider: this.settings.provider, messages }));
      else localStorage.removeItem(CHAT);
    } catch (err) {
      console.warn('agent: the conversation is not kept', err);
    }
  }

  // The conversation kept from before a reload, into the log (of the
  // provider it was with: their histories do not mix).
  restore() {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(CHAT)); } catch { /* none */ }
    if (!Array.isArray(saved?.messages) || saved.provider !== this.settings.provider) return;
    this.messages = saved.messages;
    this.settle();  // a turn cut by the reload
    const items = new Map();  // tool_use id -> its log entry
    for (const m of this.messages) {
      if (m.role === 'assistant') {
        this.show(m);
        for (const use of m.content.filter((b) => b.type === 'tool_use')) items.set(use.id, this.toolEntry(use));
      } else if (typeof m.content === 'string') {
        this.entry('user', m.content);
      } else {
        for (const r of m.content.filter((b) => b.type === 'tool_result')) {
          const e = items.get(r.tool_use_id);
          if (!e) continue;
          e.result.textContent = typeof r.content === 'string' ? r.content : r.content.filter((c) => c.type === 'text').map((c) => c.text).join('\n');
          if (r.is_error) e.item.classList.add('failed');
        }
      }
    }
    this.entry('note', '(the conversation before the reload; the world may have changed since)');
  }

  // Appends an entry to the log: kind user | assistant | thinking | tool | error | note.
  entry(kind, ...children) {
    const e = el('div', { className: `agent-${kind}` }, ...children);
    this.log.append(e);
    this.log.scrollTop = this.log.scrollHeight;
    return e;
  }

  // One request to the provider: the reply as a Claude API message.
  turn(signal) {
    const common = { system: SYSTEM, tools: this.tools.map(([def]) => def), messages: this.messages, signal };
    if (this.settings.provider === 'ollama') {
      return ollamaTurn({ ...common, url: this.settings.ollamaUrl, model: this.settings.ollamaModel, info: this.ollama.info });
    }
    return claudeTurn({ ...common, key: this.key, model: this.settings.model, effort: this.settings.effort });
  }

  async send() {
    const text = this.input.value.trim();
    if (!text || this.running || this.readiness()) return;
    this.input.value = '';
    this.entry('user', text);
    this.messages.push({ role: 'user', content: text });
    const abort = new AbortController();
    this.running = { abort };
    this.update();
    try {
      for (let step = 0; step < MAX_STEPS; step++) {
        this.note.textContent = 'thinking…';
        const msg = await this.turn(abort.signal);
        this.messages.push({ role: 'assistant', content: msg.content });  // as it came: thinking blocks go back unchanged
        this.save();
        this.show(msg);
        if (msg.stop_reason === 'refusal') { this.entry('error', 'The model declined this request.'); break; }
        if (msg.stop_reason === 'max_tokens') { this.entry('error', 'The answer was cut off (too long).'); break; }
        if (msg.stop_reason !== 'tool_use') break;
        const results = [];
        for (const use of msg.content.filter((b) => b.type === 'tool_use')) {  // in order: spawn, then move it
          results.push(await this.runTool(use, abort.signal));
        }
        this.messages.push({ role: 'user', content: results });
        this.save();
        if (abort.signal.aborted) break;
        if (step === MAX_STEPS - 1) this.entry('error', `Stopped after ${MAX_STEPS} steps.`);
      }
      this.note.textContent = '';
    } catch (err) {
      if (abort.signal.aborted) this.note.textContent = 'stopped';
      else {
        this.entry('error', err.message ?? String(err));
        this.note.textContent = '';
      }
      this.settle();
    } finally {
      this.save();
      this.running = null;
      this.update();
    }
  }

  // After a stop or an error mid-turn: a tool_use without its result (or a
  // user message without an answer) would break the next request.
  settle() {
    const last = this.messages.at(-1);
    if (last?.role === 'assistant') {
      const uses = Array.isArray(last.content) ? last.content.filter((b) => b.type === 'tool_use') : [];
      if (uses.length) this.messages.push({ role: 'user', content: uses.map((u) => ({ type: 'tool_result', tool_use_id: u.id, content: 'stopped by the user', is_error: true })) });
    } else if (last?.role === 'user' && typeof last.content === 'string') {
      this.messages.pop();
    }
  }

  // The assistant's text, thinking summaries and tool calls into the log.
  show(msg) {
    for (const b of msg.content) {
      if (b.type === 'text' && b.text.trim()) this.entry('assistant', b.text.trim());
      if (b.type === 'thinking' && b.thinking?.trim()) {
        this.entry('thinking', el('details', {}, el('summary', { textContent: 'thinking' }), el('div', { textContent: b.thinking.trim() })));
      }
    }
    const u = msg.usage;
    if (u) this.note.title = `last request: ${u.input_tokens} in (+${u.cache_read_input_tokens ?? 0} cached), ${u.output_tokens} out`;
  }

  // A tool call's entry in the log: {item, result (where its result goes)}.
  toolEntry(use) {
    const args = JSON.stringify(use.input);
    const result = el('div', { className: 'result' });
    const item = this.entry('tool', el('details', {}, el('summary', {},
      el('span', { className: 'name', textContent: use.name }), el('span', { className: 'args', textContent: args === '{}' ? '' : ` ${args}` })), result));
    return { item, result };
  }

  async runTool(use, signal) {
    const tool = this.tools.find(([def]) => def.name === use.name);
    const { item, result } = this.toolEntry(use);
    this.note.textContent = `${use.name}…`;
    try {
      if (!tool) throw new Error(`no tool ${use.name}`);
      if (signal.aborted) throw new Error('stopped');
      const out = await tool[1](use.input ?? {}, signal);
      if (out?.content) {  // a picture
        if (out.image) item.querySelector('summary').after(el('img', { src: `data:image/jpeg;base64,${out.image}`, className: 'shot' }));
        result.textContent = out.content.filter((c) => c.type === 'text').map((c) => c.text).join('\n');
        return { type: 'tool_result', tool_use_id: use.id, content: out.content };
      }
      let text = JSON.stringify(out ?? null, null, 1);
      if (text.length > RESULT_CHARS) text = `${text.slice(0, RESULT_CHARS)}… (cut)`;
      result.textContent = text;
      return { type: 'tool_result', tool_use_id: use.id, content: text };
    } catch (err) {
      item.classList.add('failed');
      result.textContent = err.message ?? String(err);
      return { type: 'tool_result', tool_use_id: use.id, content: err.message ?? String(err), is_error: true };
    }
  }
}
