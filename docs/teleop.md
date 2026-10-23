# Teleop panel and controllers

The Teleop panel (`public/js/teleop.js`, open by default) picks which robot is
driven (the newest by default, or the one double-clicked; double-clicking
anything else releases it; the others get a zero command and stand or
brake) and which of its controllers are on: **Policy**, **Base**, **IK**,
any of them or none. The choice is saved per robot file
(`mujoco-filament/controllers/<path>`); by default a robot runs the policy
made for it (GR00T on every menagerie G1 variant, no compatibility check:
switched on elsewhere, a policy reports the missing joint), Base when a
mobile base was found, IK always. Switching a policy off gives the
actuators their own gains back, and an automatic IK setup is found again
around whatever the remaining controllers drive (a G1 without its policy
gets leg chains). Keyboard: **WASD** walk / drive, **QE** turn, hold
**R**/**F** to raise or lower a locomotion policy's base height (keys typed
into a text field are ignored); the sections fold.

The **Input** select picks the keys or a **goal marker** (the default,
`public/js/goal.js`): an orange ring with a heading arrow, like the spawn
marker, that sits under the driven robot until it is dragged along the
floor (the mouse wheel or **Q**/**E** turn it while dragging, **Esc** puts
it back; a label by the pointer shows its position and heading). It is
drawn over everything, so a robot's base does not hide it. The robot then drives itself there, with the same twist command
the keys give and within their speeds, so a walking policy and a wheeled
base both follow it, in local and network mode: far off it turns towards
the goal and goes forward (a robot that cannot strafe backs up to a goal
just behind it), near it a mecanum base or a walking robot closes the rest
sideways, and at the point it turns to the goal heading (6 cm, 3.5°, with
hysteresis). A robot keeps going to its goal while another one is driven;
Reaching the goal lets it go (the readout says *arrived*; pushed or carried
off afterwards, the robot stays put). **Stop** forgets the goal, and so does Reset or driving it with the keys.

A **gamepad** (the Gamepad API's standard mapping, Xbox / PlayStation
layout; browsers report one once a button on it is pressed) drives too, in
any Input mode: its sticks walk and turn and take over from a goal like the
keys, which also do so in goal-marker mode. Two of its inputs close the
grippers of the driven robot's first and second IK chain as far as they
are pressed, like a WebXR trigger (let go: open). Its bindings are
`public/json/gamepad_defaults.json` (by default left stick walk, right
stick turn, d-pad up/down height, LT/RT the grippers); **Input → Gamepad**
(the only choice for a robot that does not walk but has IK chains) shows
them per action, **Set** binds the next button pressed or stick pushed,
changes are kept in the browser, **Defaults** goes back to the file and
**Export** copies them as JSON for it.
In network mode a robot has one goal for everyone: the server follows it
(`server/goal.py`), every client shows it, and the last one set wins.

Goals can also come from a script, for several robots at once:

- **In the page** (devtools console or page automation, local and network
  mode): `goals.move(which, {forward, left, turn})` sets each robot's goal
  relative to its own pose (m, rad), `goals.set(which, {x, y, yaw})` an
  absolute one, `goals.clear(which)` stops them, `await goals.wait(which)`
  resolves once all have arrived, `goals.list()` shows names, poses and
  goals. `which` is `'all'` (every robot whose walking policy or base is
  on), a name (`'r3'`), a label, or a list of those. E.g.
  `goals.move('all', {forward: 1}); await goals.wait('all'); goals.move('all', {forward: -1})`.
  Behaviours on a timer (`public/js/swarm.js`):
  `goals.randomWalk({model: 'microduck', exclude: ['r1']})` gives a few
  robots at a time a random goal nearby, inside a play area;
  `goals.chase({target: 'r1', distance: 0.4})` has them walk after robot
  `r1` and stop short of it, facing it. Both take `robots` (as `which`),
  `model` (part of the label) and `exclude`; each returns an id,
  `goals.running()` lists them, `goals.stop(id)` (or `goals.stop()` for all)
  stops them and clears their goals. A robot two behaviours take follows the
  one started last; the other options are at the top of `swarm.js`.
- **On the server** (HTTP): `GET /api/robots` lists the robots
  (`name`, `pose`, `goal`, `drives`, `arrived`); `POST /api/goal` with JSON
  `{robots, forward, left, turn}` or `{robots, goal: {x, y, yaw}}` sets
  goals, `{robots}` alone clears them. `scripts/goal_demo.py` drives robots
  back and forth with it:
  `python3 scripts/goal_demo.py --robots r1,r2 --distance 2 --times 3`.

A robot that is being driven with the keys in the Teleop panel ignores its
goal, as with the marker.

The controllers themselves: [policies](policies.md), [mobile bases](mobile-bases.md),
[IK chains](ik.md); and the panel's [ROS bridge](ros.md).
