# CRAM in the lobby: plan of what to verify

Goal: from the lobby, start a separate Binder session with CRAM (cramera's image, the same
`/shared/mjcf_assets`), connect to it over HTTP/WebSocket, build and keep CRAM's world
(semantic digital twin, SDT) in step with the lobby's, and write annotations and CRAM
plans in the Behavior editor, IPython style, to drive the lobby's robots.

Status: partly verified (2026-11-18) in a Binder session of cramera main 22487ea (CRAM
pinned at d938a05, mujoco 3.11.0); see "Results" below. The plan after it is the
original list, read against CRAM upstream 64d8c20.

## First milestone: done (2026-11-18)

The Behavior example "Panda pick & place by CRAM" (`public/behaviors/cram_panda_pick.py`):
the page starts (or reuses: `?cram=<session URL?token=…>`, else the one this browser
used last) a cramera Binder session, writes a `cram` kernelspec there, starts the kernel
and runs `public/js/cram/lobby_cram.py` in it (`public/js/cram/cram.js`; the session's
REST API and kernel WebSocket answer the page's origin). The tree sends one cell,
`pick_and_place(...)` with the Panda's (menagerie `panda.xml`) base pose and joints and
the cube's pose from the lobby; CRAM builds its world from them (generated robot class,
TCP body at the hand + 0.1034 m), runs park → PickUp → Place → park kinematically and
returns the actuator targets every 0.05 s (188 steps), which the tree plays back with
the IK off (`lobby.set_actuators`). Headless test: the cube lifted to 11.7 cm, set down
~1 cm from the target, pushed to 2.3 cm off by the fingers on the way back to park.
From a fresh page ~1 min to the end (session 10 s, CRAM import 12 s, first plan 19 s,
later plans 1.5 s). Open loop; one robot in `lobby_cram.ROBOTS`; tested in local mode only.

## Closed loop, per action: done (2026-11-18)

The example now runs closed loop (`lobby_cram.start` once, then `lobby_cram.act` per
action). CRAM's world is kept; before each action (park, pick, place) the tree sends the
lobby's joints, the cube's full pose and whether the Panda holds it. CRAM syncs them (a
held cube's pose under the TCP too; the fingers are not synced: CRAM's are the gripper's
command), rebuilds its world when it holds the cube and the lobby does not, and plans that
one action. The tree plays it back and watches it: the cube moved before the grasp, or
slipped out while carried, aborts the action; the outcome is checked (held after a pick,
within 2 cm of the target after a place); a failed pick is retried after a park, a failed
place restarts at the pick.

What it took to make the physics follow (each found in headless runs):
- sample the plan once per executor tick (`ex.tick`), not on every `notify_state_change`:
  the plan applies states while it tries things (the gripper flickered open mid-carry);
- a floor in CRAM's world (else a plan may press the hand into the lobby's floor);
- a cube told to CRAM in its most upright, least-turned equivalent pose (its one grasp
  candidate otherwise asks for the wrist beyond joint 7's range);
- `grasp_height` 0.045 for the 5 cm cube (CRAM brings the TCP 1.8 cm below that point:
  0.025 grasped at the floor);
- the playback: targets ramped linearly between ticks in the lobby (`set_actuators(...,
  ramp)`, drive.js `ramp`; on a server `joint_cmd`'s `ramp`, server.py `Robot.override`;
  the server side untested), one tick ahead, waiting whenever the arm lags > 0.05 rad,
  before a gripper change until the arm is within 0.008 rad, and after it until the
  fingers stop (the menagerie Panda's gripper closes slowly and squeezes with ~2.5 N);
- noslip 10, pyramidal cone (as the SO-ARM101 example).

Headless tests (all succeed): undisturbed 0.4 cm from the target; the cube moved 10 cm
during the pick's approach (aborted, replanned) 0.5 cm; the cube taken out of the fingers
while carried (aborted, picked again where it lay) 0.5 cm; cube spawned at yaw 0.7 / −1.2
rad 0.6 / 0.2 cm. Binder sessions are culled after ~20 min idle: `cram.js` checks the
saved one and launches anew.

## Results (2026-11-18)

Test scripts and raw output: the session scratchpad of 2026-11-17/18 (`lobby_pick*.py`,
`generic_robot.py`, `fr3_pick.py`, `fr3_traj.py`, `kernel_test.py`, `lobby_play.mjs`,
`sweep*.jsonl`).

**The whole chain works for one robot.** Tracy and FR3 + hand built from the lobby's
MJCF, a cube spawned from the lobby's `cube.xml`, a CRAM plan (park, `PickUpAction`,
`PlaceAction`, park) in `KinematicallySimulatedPlanExecutor`, sent as cells to a Jupyter
kernel in the session, and the joint trajectory replayed on the lobby's FR3 in physics:
the cube ends 1.6 cm from CRAM's target (open loop, no feedback).

| Item | Result |
| --- | --- |
| A1 parsing (mujoco 3.11 + SDT `MJCFParser`) | 315 of 415 assets load: objects 167/168, robots 73/119, environments 75/128. Failures: 39 have capsule geoms (`NotImplementedError: mjGEOM_CAPSULE`, GARMI among them); 40 use syntax mujoco 3.11 does not know (`unrecognized attribute`, `invalid keyword`; mujoco 4.0 loads 411/415); 15 have several joints on one body, as the aicor mobile bases (`base_x/base_y/yaw`: PR2, Tiago, HSR, Armar7, donbot, agibot G2), `No world entity 'base_y'`; 6 reference missing files. Parse time 0.4 s median; the biggest environments 10 s–14 min and up to 3.5 GB. |
| A2 positions | mujoco 3.11 and 4.0 agree on all 22,776 body positions where both load. SDT vs mujoco: 3,105 of 20,179 off by >1 mm, all in scenes that rotate with `euler` (the supermarket, every robocasa kitchen): **SDT's parser reads `MjsBody.quat` / `frame.quat` only, and MjSpec keeps `euler`/`axisangle`/`xyaxes` in `.alt` with `quat` left at identity**, so those rotations are dropped (an upstream fix: resolve `.alt` to a quaternion). |
| A3 spawn / remove at run time | `merge_world_at_pose` of a parsed asset 0.03–0.04 s, `remove_branch_from_world` 0.016 s; plans after a spawn and after a removal work, with new executors. Spawning while a plan runs: not tried. |
| A4 state rate | Tracy + GARMI apartment (137 bodies, 145 DoFs): writing joint values 0.1 ms, writing + reading every body's world pose 7.7 ms per update: fine for 30 Hz. |
| A5 grasp ownership | In the kinematic executor CRAM moves the held cube itself; in the lobby the cube follows contact. Replayed open loop the two diverged by 1.6 cm; a real executor needs the lobby's state back (closed loop). |
| B6 robot classes | CRAM's own classes apply to the lobby's MJCF only where names match: Tracy, iCub3 yes; Stretch, G1 no (menagerie names); PR2, Tiago, HSR, Armar7, GARMI fail in the parser. A **generated** class from ~15 lines of data (arm root/tip, TCP site turned into a body, finger joints with open/closed values, approach/closing axis, park pose: what the lobby's IK chain holds) made FR3 + hand pick and place twice, exactly on target. |
| Start state | Tracy's first plan fails (`MoveGripper/width made no progress`): the MJCF leaves the Robotiq knuckles at 0, at a joint limit; set the gripper's open state (and park) before the first plan. |
| Mimic joints | The FR3 fingers are coupled by an equality in the MJCF, independent in SDT (the parser does not read the equality); the trajectory had the fingers at 0.0 / 0.04. Map them back (one tendon actuator in the lobby). |
| C9 trajectory | The kinematic executor runs a pick-and-place in ~1.2 s wall time; its state changes are 0.05 s control steps (joints at exactly the 0.5 rad/s limit), 217 samples = 10.9 s of motion: stream them paced by that step. |
| D12 kernel | A kernelspec on cramera's venv (`ipykernel` 6.30 is in it) with ROS sourced; started over REST, driven over the kernel WebSocket with `Origin: https://yxzhan.github.io` (accepted). Setup cell 10.6 s, plan cells 1.1–1.4 s in the same live world, kernel 0.75 GB. Without ROS sourced, `coraplex.plans.executors` fails to import (`json_msgs`). |
| D14 session | `/shared/mjcf_assets` mounted; pod reaches github.io, GitHub, the hub. BinderHub's build API answers any origin (`*`); the session's API echoes the lobby's origin on GET, but its OPTIONS preflight had no allow-origin: check a POST with a JSON body from the real page. Pods are capped at 16 CPUs / 32 GB (`nproc` says 240); a session was stopped by the hub after ~2 h. |
| Image | No robot description packages (`package://iai_*` unresolved), so CRAM's usual `from_description` (URDF) path is unavailable; `griplink_interfaces` missing (Daisy). |

## What the code already says

- **cramera is CRAM → viewer.** `cramera_bridge` publishes a running CRAM world to the
  viewer and records it; the plan builder composes a plan in the browser, generates a
  demo and runs it as a child process, whose world it builds itself (an environment
  bundle + robot + objects, `cramera_bridge/plans/world.py`). Nothing takes a world
  from outside and keeps following it. The parts worth reusing: the plan protocol
  (`cramera_protocol.plan_builder`, format 1.4), the action catalog read from the CRAM
  workspace (`plans/workspace.py`), grounding / validation / run log, the prepared
  interpreter (`process_sessions.py`, `prepared_demo.py`: CRAM imported once, plans run
  in it, which is the IPython model).
- **SDT can change at run time.** `world.modify_world()` adds and removes bodies,
  connections and annotations; `WorldSynchronizer` sends model changes and state
  between CRAM processes (ROS, `std_msgs/String`), `FetchWorldServer` hands a whole
  world to a new process. `MJCFParser.from_file/from_xml_string` reads MJCF through
  `MjSpec`; `merge_world_at_pose` puts one world into another.
- **The lobby's world is a recipe.** `server.py` builds it as base + `spec.attach(asset,
  frame, prefix)` per spawned asset; the page knows the same list (rel, prefix, pose)
  and every body's pose and joint value each step.
- **CRAM robots are hand-written.** `semantic_digital_twin/robots/`: ~17 classes (PR2,
  Tiago, HSR, Stretch, G1, Tracy, GARMI, Justin, Armar7, iCub3, …), each a few hundred
  lines naming arms, grippers (approach and closing axis), TCP, cameras, torso, base,
  joint states, controllers. `MinimalRobot.from_branch_in_world` takes any branch, but
  has no arms or grippers, so no Pick/Place. The lobby has 119 robots; it already finds
  IK chains in a model (root, tip, TCP site, gripper actuators with open/closed values;
  `ik_defaults.json` for 12 robots with hand-tuned ones).
- **Annotations can be inferred.** `WorldReasoner(world).infer_semantic_annotations()`
  (rules: drawers, doors, handles, …), used by CRAM's GARMI demo.
- **Versions differ.** The lobby runs mujoco 4.0.0, CRAM pins mujoco 3.11.0.

## Proposed architecture (to be checked)

```
lobby page / server ──WS── CRAM session (cramera image + a "lobby link" process)
  scene recipe, spawn/remove ──►  SDT world: base + MJCFParser(asset) per spawn
  body poses, joint values   ──►  world.state (free bodies, joints)
  joint targets / velocities ◄──  plan executor ("LobbyPlanExecutor")
  editor cell (annotations,  ──►  run in the prepared interpreter, plan run log back
  plan code)
```

- The CRAM side holds the SDT world; the lobby stays the physics (like a real robot).
- Execution: either giskard → ROS → rosbridge → lobby (what the GARMI and Tracy
  workshop demos do, one bridge script per robot), or a new executor that sends joint
  targets over the same WebSocket and reads the state back (no ROS needed).

## To verify

### A. World build and sync

1. **One asset, parsed.** In the CRAM image, `MJCFParser.from_file` on a range of
   `mjcf_assets` models (a table, a kitchen with drawers, 10 robots from different
   sources). Record which fail and why (mujoco 3.11 vs files written for 4.0; meshes,
   `include`, plugins, `frame`, flex). *Check:* a script over the asset list, a table of
   pass/fail.
2. **Composing like the lobby.** Build base + several assets with prefixes and poses
   via `merge_world_at_pose`; compare body names and world poses with the lobby's
   `app.data.xpos` for the same scene. Names must map 1:1 (prefixes, the lobby's
   `r1/…`). *Check:* diff of poses below 1 mm.
3. **Spawn and remove at run time.** Add and remove an asset inside `modify_world()`
   while a plan executor exists: do the executor, the collision checker and giskard's
   world follow, or does a running plan need to stop first? *Check:* spawn a cube while
   a simulated plan is idle, then run a Pick on it; remove a robot that is not in use.
4. **State rate.** Push the lobby's state (free bodies + joints) into `world.state` at
   10–30 Hz. Cost per update for a big scene (apartment + 2 robots); whether every
   update triggers expensive callbacks (collision, TF, viz). The /tf flood that stalled
   giskard (2026-11-17 test) is a warning. *Check:* timing of `world.state` updates
   with the callbacks a plan run attaches.
5. **Who owns which state.** Lobby → CRAM for everything physics moves; CRAM → lobby
   only for the commands. Grasped objects: CRAM attaches them to the gripper (a
   connection change), the lobby holds them by contact/weld. *Check:* do the two
   disagree after a Pick, and which one the next step believes.

### B. Robots

6. **A generated robot annotation.** Write a generic `AbstractRobot` subclass built
   from the lobby's IK chains: arm = chain root..tip, manipulator = TCP site + gripper
   actuators (open/closed), base = free joint or the planar base the lobby drives.
   Missing: approach and closing axis (guess from the TCP frame, or a small per-robot
   override in `ik_defaults.json`), cameras, torso, collision rules, velocity limits.
   *Check:* Pick and Place with it on 3 robots that CRAM also has (Tiago or PR2, Stretch,
   G1) and compare with the hand-written class; then 3 robots CRAM lacks (SO-ARM101,
   FR3 alone, a mobile manipulator).
7. **What the actions need from a robot.** Read which robot parts coraplex's
   Pick/Place/Navigate/ParkArms/OpenDrawer use (arms, grippers, base type, torso, park
   positions, `get_left_arm_if_specified`, …), so the generated class covers exactly
   that. *Check:* list per action.
8. **Robots without arms** (quadrupeds, wheeled bases, humanoids driven by a policy):
   Navigate only? Decide whether they are in scope.

### C. Execution in the lobby

9. **Executor choice.** (a) giskard → ROS → lobby: works today for GARMI/Tracy/FR3 with
   one bridge per robot; check it can be made generic from the lobby's ROS profiles.
   (b) A `LobbyPlanExecutor`: giskard (or the kinematic executor's motions) produces
   joint trajectories / velocities, sent over the WebSocket as joint commands; the
   lobby already integrates velocity commands. *Check:* a ParkArms and a Pick on one
   robot with each, how far the robot's motion matches CRAM's expectation.
10. **Grasping.** A gripper closing in physics may slip or miss where CRAM's kinematic
    world says the object is held. *Check:* how the plan notices (perception again,
    the run's grounding anew, the lobby's contact state).
11. **Perception.** cramera's plans detect before a grasp (RoboKudo). In the lobby,
    the "detection" can be the ground truth from the synced world, or the lobby's
    cameras. *Check:* whether cramera's lenient check can run without RoboKudo
    (`f1f58fc` says so) and use the world's poses.

### D. Editor and session

12. **Prepared interpreter.** Can cramera's `process_sessions` / `prepared_demo` serve
    as a kernel: a long-lived process with CRAM imported and the synced world, running
    code cells (annotations, plan definitions, `run(plan)`) sent from the editor, with
    stdout, errors and the run log coming back? Or simply a Jupyter kernel through the
    Binder session's own API (`binder_session.py` already drives terminals; the
    kernel WebSocket is similar). *Check:* start time, memory, one cell round trip.
13. **What the editor writes.** Python against CRAM's API directly (flexible, long
    import names, keyword-only API that moves with upstream: see the GARMI demo
    breaking on 96df669a0), or cramera's plan document (format 1.4, steps with a skill
    and fields, validated, stable). Probably both: plan documents for the common cases,
    raw cells for the rest. *Check:* write the so101_banana pick & place both ways.
14. **Starting the session from the lobby.** BinderHub's build API from the page (CORS?
    token handling), one session per lobby room or per user, time to start (the
    cramera image is big; about 1 min when cached), and whether the session can open a
    WebSocket back to the lobby's server (outbound from the Binder pod) or the lobby
    connects to the session's proxied port (`/proxy/<port>/`, as with rosbridge).
    `/shared/mjcf_assets` is mounted in every session on that hub: check it is in the
    cramera image's sessions too.
15. **Several robots, several plans.** One world for the whole lobby with several
    robots, plans for each robot at the same time, or one plan at a time? cramera
    allows one run at a time.

### E. Upstream risk

16. CRAM moves fast (keyword-only refactor, renamed fields, the `Match` JSON bug that
    stopped giskard on 2026-11-17). Pin a CRAM commit per image, as cramera's binder
    branch does (`CRAM_REF`), and keep the lobby-specific code in one package.

## Suggested order

1 → 2 → 7 → 6 (one known robot) → 9b (ParkArms) → 4/5 → 3 → 12 → 13 → 14 → the rest.
The first milestone: a cell in the editor makes one robot in the lobby pick up a
spawned cube, with CRAM's world built from the lobby's recipe.
