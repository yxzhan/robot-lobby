# Shared world (Python server)

```sh
pip install mujoco numpy aiohttp onnxruntime
npm run server   # python3 server/server.py 4007
```

`server/server.py` steps one MuJoCo world in Python and serves `public/`
too; every browser that opens it (http://localhost:4007, or through the
JupyterHub proxy) sees and changes the same world. The page notices the
server (`server.json`) and switches to network mode (`public/js/net.js`), in
which it loads no MuJoCo at all:

- **Drawing.** The server compiles each model file once and serves what is
  drawn at `/render?rel=…` (`server/render_asset.py`): visible geoms,
  materials, the meshes they use (compiled: MuJoCo recenters meshes), the
  pixels of MuJoCo's builtin textures, image textures by path, and the
  kinematic tree. Gzipped, cached by the browser (ETag); a G1 is ~7 MB. The
  arrays keep MjModel's field names, so `buildVisual` turns them into
  Filament parts like it does a WASM model.
- **State.** 30 times a second the server sends qpos and mocap poses as
  float32 (ctrl at 5 Hz, for the panels): ten G1s are ~1.7 KB per state,
  ~50 KB/s. The page interpolates between states, a little behind (see
  the network meter; quaternions on the short arc), and places the bodies itself
  (`forwardKinematics`, MuJoCo's `mj_kinematics`, matching it to 1e-15).
- **Requests.** Spawn, Clear / ✕, Reset, Pause, the Stage panel's
  poses and sliders, Shift+drag, double-click picking (the server casts the
  ray) and the World panel's options go to the server, which applies them
  and tells every browser. Robots are attached in the same order on both
  sides, so bodies, joints and actuators are numbered the same.
- **Controllers on the server.** Policies (`server/policies.py`: GR00T
  and Playground policies on onnxruntime, same ids as `policy.js`), mobile
  bases (`server/base.py`) and IK (`server/ik.py`) are ports of `groot.js`,
  `playground.js`, `base.js` and `ik.js`,
  ticked at 50 Hz with the physics; ten G1s walking under GR00T keep real
  time. The Teleop panel works as in local mode through `public/js/remote.js`.
  The keys' drive command goes to the server when it changes and is
  repeated while held; the server stops a command not repeated for 1 s
  (a closed or frozen tab), and the base's measured twist comes back for
  the readout. Base setups saved from the panel go to
  `server/saved/base.json`; `public/json/base_defaults.json` holds the built-in ones.
- **IK markers.** They work as in local mode:
  dragging a marker moves it at once and sends its target (in the chain's
  reference body frame, so it rides with the robot); the server sends
  targets and tracking errors back 5 times a second, so every browser sees
  the markers move. Chain edits, Save (to `server/saved/ik.json`, for
  everyone) and Defaults are requests too. `public/json/ik_defaults.json` holds the
  built-in setups for both sides.
- **Stages.** Saved on the server for everyone, in
  `mjcf_assets/saved_stage/<id>/` (see spawning.md): the server writes the
  world's items, state and controller setups, the page adds the name,
  picture, view and camera setups (`/api/stages`). Loading one replaces
  the world for everyone (`stage_load`); `?stage=<id>` asks first.
- **Network meter.** Bottom right, next to the frame meter: round trip
  (ping, includes the page's own frame time), states per second, download
  rate, and how far the view trails the newest state: one state interval
  plus three times the arrival jitter (40–250 ms).
- **Reconnecting.** When the connection drops (the server stopped, Wi-Fi
  changed, a phone slept: pings unanswered for 5 s count too), a banner
  says so and the page tries again after 1, 2, 4, 8, then every 10 s, at
  once on Retry now, when the network comes back or the tab is shown
  again. The server sends its world on reconnecting; a restarted server's
  (empty) world replaces the old one.
- **Memory.** Copies of a model share its meshes in the server's world:
  ten G1s compile to ~75 MB instead of ~750 MB.
- **CPU.** One world steps on one thread of its own, apart from the event
  loop that talks to the browsers, with MuJoCo's thread pool solving the
  robots' constraint islands side by side (up to 8 threads, fewer when the
  container's CPU quota is lower; `PHYSICS_THREADS=n` sets it). Same result
  as one thread, bit for bit. The controllers tick one after the other:
  they are mostly Python, which threads would only take turns at. Twelve
  G1s walking under GR00T keep real time (eight before). More worlds are
  more servers, e.g. one per Binder pod.

Not yet in network mode: the World panel's default geom properties. The scene lives as long as
the server process. Local mode keeps everything in the browser as before.

`?local` in the URL keeps the page fully local (WASM physics) even when the
server serves it.
