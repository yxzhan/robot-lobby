# Control between several clients (design)

Status: steps 1–3 built (`server/control.py`, `public/js/net/control.js`,
the Teleop and Share panels); see "Built" at the end for what differs from
the design. Applies to server mode (`server/server.py`); local mode has one
client and does not change.

## Problem

Every browser on a shared world can send every op for every robot
(`server/server.py` `handle_op`). Nothing says who is driving what:

- Two people on the Teleop panel driving one robot fight: their `cmd`s
  overwrite each other at 4 Hz (`REPEAT_MS`), their IK markers jump, one
  switches the policy off under the other (`use`).
- Anyone can `reset`, `pause`, `remove`, `stage_load`: the whole world
  changes for everyone, and nobody sees who did it.
- The server has no idea who a connection is: a `ws` and nothing else. A
  reload is a new client.

The one existing rule is `ros_hold` (the first browser to switch a robot's
ROS on publishes it; let go when it leaves). This design generalizes it.

## Goals

- One person drives a robot at a time, and everyone sees who.
- Using the page alone stays as it is: no extra clicks before driving.
- A tab that closes, sleeps or loses its network does not keep a robot
  forever; a reload does not lose it either.
- The server enforces the rules (a page that skips them gets an error),
  the page shows them (controls greyed out, badges).

Not goals (now): accounts, passwords, per-person permissions. Everyone with
the session's link (its token) is trusted not to be malicious; this is
about coordination, not security.

## Concepts

### Clients

A client is a browser tab (or a script) with:

- `id`: random, made once per browser and kept in localStorage
  (`robot-lobby/client-id`); a second tab of the same browser adds a
  suffix kept in sessionStorage, so two tabs are two clients.
- `name`: typed in the Share panel or the people row (default: its kind
  of device and a number, "PC 1", "Mobile 2", "VR 1"), kept in
  localStorage; `color`: given by the server from a palette, unique among
  those connected.

The page sends `hello {id, name}` first thing after connecting. The server
keeps a `Client` per id (`world.clients` becomes ws → Client) and tells
everyone the list: `{type: 'clients', you, clients: [{id, name, color,
idle}]}` on join, leave, rename.

A client that drops keeps its holds for a grace time (`GRACE` 30 s): a
reload, a phone waking up or Wi-Fi changing comes back with the same id and
finds its robots still held. After that, its holds are freed.

The HTTP API (`/api/goal`) is the client `api` (name "Script").

### Holds

A robot is *free* or *held* by one client. Held: only its holder may send
it robot ops (below); everyone else gets an error. Free: anyone may, and the
first robot op from someone **claims it implicitly** — the person alone
just drives, as now, and a second person sees "held by Alice" the moment
Alice starts.

A hold ends when:
- the holder lets go (Release button, or `hold {id, on: false}`),
- the holder has been idle on that robot for `IDLE` (5 s without a robot
  op for it; held keys and dragged markers count, a standing robot does not),
- the holder is gone for longer than `GRACE`,
- the robot is removed or the world replaced (`stage_load`, `restart`).

Taking a held robot: only once its holder has been idle on it for
`IDLE_TAKE` (5 s): *Take over* takes it at once, the old holder is told.
An active holder keeps it (there is no asking).

A robot's hold is in the scene items (`item.holder: client id or null`,
`item.held_since`), so it reaches everyone in `scene` / `meta` messages and
a joining page.

`ros_hold` becomes part of this: switching a robot's ROS on needs its hold
(it is a robot op), and holding a robot implies its ROS is published by the
holder's browser if that browser has ROS on. One rule instead of two. The
`/clock` hold stays as it is (it is not a robot's).

### Op classes

Server-side table, one line per op, checked in `handle` before
`handle_op`:

| class | ops | rule |
|---|---|---|
| robot | `use`, `cmd`, `command`, `skill`, `goal`, `ik_config`, `ik_target`, `ik_reset_target`, `ik_gripper`, `base_config`, `joint_cmd`, `reset_robot`, `move`, `place`, `ros_hold` (a robot's) | needs the hold, or claims a free robot |
| robot by index | `ctrl {a}`, `qpos {adr}` | the robot that owns actuator `a` / qpos `adr` (`state_layout`), then as above |
| robot removal | `remove {ids}` | each one free or held by the sender |
| touch | `perturb`, `place_body` | anyone (pushing a robot with the mouse is allowed; it is physics, not driving) |
| world | `spawn`, `reset`, `pause`, `physics`, `restart`, `stage_load` | anyone, as now |
| files | `ik_save`, `ik_defaults`, `base_save`, `base_defaults` | the robot's holder (they write server/saved/ for every robot of that file) |
| read | `ray`, `ping`, `dynamics`, `hello`, `hold` | anyone |

A refused op answers `{type: 'error', op, code: 'held', id, holder, msg}`
(also without `rid`: today an error goes back only as text, so the page
can show it instead of a robot just not moving).

`cmd` from a client that is not the holder is dropped *silently after the
first error* (held keys repeat 4×/s; one toast is enough).

### World ops

Stay as they are: anyone may reset, pause, change physics, spawn or load a
stage, with no announcement or confirmation.

### Agents and behavior trees

The Agent panel and the Behavior panel run in a page and send that page's
ops: they act as their client, under its holds. A behavior tree driving a
robot keeps it held (its ops count as activity).

## Page

- **Teleop panel**: the robot picker shows each robot's holder (color dot +
  name, "you", or nothing when free). For a robot held by another: the
  controls are greyed, a line "Held by Bob" with *Take over* (enabled once
  Bob is idle); for one held by you: *Release*.
- **In the scene**: a small label with the holder's name and color above a
  held robot (toggle in the Render panel, on by default when more than one
  client is connected).
- **Share panel**: your name (editable), who is connected (name, color,
  idle).
- **Toasts**: refused op, hold taken over.
- Input routing (`drive.js`): keys, gamepad and XR go only to a robot this
  client may drive; choosing a robot held by another in the picker does
  not send anything until the hold is yours.

## Protocol changes (summary)

Client → server:
- `hello {id, name}`; `rename {name}`
- `hold {id, on}`; `take {id}` (idle holder only)

Server → client:
- `{type: 'clients', you, clients: [{id, name, color, idle}]}`
- `item.holder`, `item.held_since` in `scene` / `meta` items
- `{type: 'hold', id, holder, by: 'claim'|'release'|'idle'|'gone'|'take'}`
- `error` with `code: 'held'`

`ros` messages (`holds: you | other`) stay for `/clock`; robots' come from
`item.holder`.

## Server internals

- `Client` dataclass: id, name, color, ws (None while gone), gone_at,
  last_seen per robot.
- `World.holders: {robot id: client id}`; `World.claim(client, r)` →
  ok / raises `Held`.
- `World.owner_of_actuator(a)`, `owner_of_qpos(adr)` from `state_layout`
  (built once per scene in `swap`).
- A once-a-second task frees idle and gone holds and broadcasts changes
  (the run loop already ticks; or `expire_commands`' caller).
- `cmd_at` keys stay (their ws becomes the client), so a gone client's
  commands still stop after `CMD_TIMEOUT`, before its hold expires.
- `/api/robots` shows `holder`; `/api/goal` on a robot held by someone else
  is a 409 unless `{"force": true}`.

## Steps

1. Identity: `hello`, `Client`, the clients list, the name in the Server
   panel, grace on reconnect. Nothing refused yet.
2. Holds on the server: implicit claim, the op table, release on idle /
   gone, `error code: 'held'`; `ros_hold` folded in. The page shows refusals
   as toasts.
3. Page: holder badges in the Teleop picker and in the scene, greyed
   controls, Release / Take over.
4. (Later) teams for multiplayer tasks
   (TODOS.md "Tasks and challenges"): a hold given to a team.

Test: two (three) headless clients on one server (`docs/server.md` setup):
A drives, B's `cmd` refused; A idles, B takes over; A reloads within grace
and keeps its robot; A's tab closes, its robot is free after `GRACE`.

## Decisions (2026-10-30)

- Free robots are claimed implicitly by their first robot op.
- Take over only from an idle holder; no asking.
- World ops stay open, no announcement (as now).
- Anyone may push (perturb) a held robot.

## Built (2026-10-30)

- Server: `server/control.py` (clients, holds, the op check), wired into
  `server/server.py` (`handle`, `op_robots`, `sync_control`, the WebSocket
  handler, `/api/robots` `holder`, `/api/goal` 409 / `force`).
- Page: `public/js/net/control.js` (hello, clients, holds); the Teleop
  panel's hold line (Free / Yours + Release / Held by … + Take over) and
  the rest of the panel locked (inert) for a robot someone else holds; the
  picker names each robot's holder; the Share panel lists people (in
  server mode), the robots they drive, and this page's name. Drive, goal,
  IK and joint commands to a robot held by someone else are not sent.
- Holds only for robots with controllers (`Robot.key`); objects and
  environments stay open to everyone.
- `ros_hold` stays as it was (who publishes a robot's ROS), not folded into
  the holds; ROS joint commands and cmd_vel of the page are robot ops and
  need the hold like the rest.
- A robot walking to its holder's goal counts as busy (not idle).
- Each client's device ("iPhone · Safari", "Quest · Browser", "Windows ·
  Edge", "script") from the User-Agent of its WebSocket handshake
  (`control.device`), shown in the people list and the Take over tooltip;
  the page's `hello` says whether it has a touch screen, since iPadOS
  Safari presents itself as a Mac.
- Over the view (`public/js/net/presence.js`): a tag over every held
  robot (its holder's color and name, "you"; faded once the holder has
  left it alone 2 s), the people row under the toolbar's status (this
  page first, click it to rename; tooltips with device and robots), and
  notices for 4 s: someone took over a robot this page drove, someone
  drives the robot this page's Teleop panel shows, joined, left, renamed.
- A robot someone else spawns is not selected in Teleop; one this page
  spawns is, and its spawner holds it. Picked by itself (on joining, or
  its robot gone), Teleop takes the newest robot this page holds, else
  the newest free one, else none.
- Tested with two WebSocket clients and two headless pages: claim, refusal,
  a quiet stop, reload within grace, release, take over after idle,
  free 30 s after the holder left, the API.
