// Policy commands other than the drive twist: what a policy takes as input
// besides its observation of the robot (a head pose, a base height, a gait),
// declared by the policy so the Teleop panel, the server and the
// observation can handle any of them without code of their own.
//
// A controller that takes some has
//   commands  {name: definition}, in the order the panel shows them
//   command   {name: value}, the current values (commandDefaults to start)
// A definition (in a policy.json's "commands", or in code like GR00T's):
//   type     'scalar' | 'vector' | 'enum' | 'bool'
//   label    shown in the panel (default: the name)
//   about    one line: what it does (the tooltip)
//   range    scalar: [lo, hi]; vector: one [lo, hi] per dimension
//   dims     vector: the dimensions' names
//   unit     shown after the value; a vector may have one per dimension
//   values   enum: the choices
//   default  the value it starts with (zero, the first choice or false if
//            left out)
// Any other type is shown as a JSON text field and passed on as it is.
//
// In an observation a command is the part `cmd:<name>` (commandVector): a
// scalar one number, a vector its numbers, a bool 0 or 1, an enum one-hot
// over its values. The drive twist stays `cmd` = [vx, vy, yaw rate] with
// the key speeds `drive` (policy.js), which the keys, the goal marker and
// ROS cmd_vel set; a command named `height` also follows the R/F keys.
// server/policies.py (commands_of, clamp_command, command_vector) is the
// same in Python.

export const COMMAND_TYPES = ['scalar', 'vector', 'enum', 'bool'];

const clamp = (v, [lo, hi] = [-Infinity, Infinity]) => Math.min(hi, Math.max(lo, v));

// The value a command starts with.
export function commandDefault(def) {
  if (def.default !== undefined) return structuredClone(def.default);
  switch (def.type) {
    case 'scalar': return clamp(0, def.range);
    case 'vector': return (def.range ?? def.dims ?? []).map((_, i) => clamp(0, def.range?.[i]));
    case 'enum': return def.values?.[0] ?? '';
    case 'bool': return false;
    default: return null;
  }
}

export const commandDefaults = (defs) => Object.fromEntries(Object.entries(defs ?? {}).map(([k, d]) => [k, commandDefault(d)]));

// `v` made a valid value of the command (within its range, one of its
// values); throws if it cannot be one.
export function clampCommand(def, v) {
  const num = (x) => {
    const n = Number(x);
    if (!Number.isFinite(n)) throw new Error(`not a number: ${x}`);
    return n;
  };
  switch (def.type) {
    case 'scalar': return clamp(num(v), def.range);
    case 'vector': {
      const n = commandDefault(def).length;
      if (!Array.isArray(v) || v.length !== n) throw new Error(`expected ${n} numbers`);
      return v.map((x, i) => clamp(num(x), def.range?.[i]));
    }
    case 'enum':
      if (!def.values.includes(v)) throw new Error(`not one of ${def.values.join(', ')}`);
      return v;
    case 'bool': return !!v;
    default: return v;
  }
}

// The command's part of an observation (cmd:<name>).
export function commandVector(def, v) {
  switch (def.type) {
    case 'scalar': return [v];
    case 'vector': return v;
    case 'enum': return def.values.map((x) => (x === v ? 1 : 0));
    case 'bool': return [v ? 1 : 0];
    default: throw new Error(`a ${def.type} command cannot be observed`);
  }
}

// Sets command `name` of controller `c` (a remote one sends it to the
// server); returns the value set.
export function setCommand(c, name, v) {
  const def = c.commands?.[name];
  if (!def) throw new Error(`${c.name}: no command ${name}`);
  const value = clampCommand(def, v);
  if (c.setCommand) c.setCommand(name, value);
  else c.command[name] = value;
  return value;
}
