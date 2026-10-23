
# Robot Lobby

An multiplayer MuJoCo sandbox in the browser: spawn robots,
drive them, pull them around, in one physics world shared by everyone
connected (a Python server, e.g. on Binder), or alone with the physics in
the page.

**Live demo:** https://yxzhan.github.io/robot-lobby/ (front end only: the
physics runs in your browser, with part of the model library).



https://github.com/user-attachments/assets/242248f9-c4d9-415c-8fe0-4d691a56f08c



Physics from the official MuJoCo WASM bindings
(`@mujoco/mujoco`), rendering with Filament.js (WebGL2, PBR, sun shadows, SSAO).
No bundler — `public/` is a static site.

## UI Guide

- **Camera:** left-drag orbits, right-drag pans, the wheel zooms. Click an
  axis of the gizmo (bottom left) to look along it. The meters at the
  bottom right show the frame rate (click to cycle) and the sim speed.
- **Spawning:** drag a model from the **Assets** panel onto the floor. While
  dragging, the wheel or Q/E turns it, Shift+wheel or R/F raises it, Esc
  cancels.
- **Selecting and pulling:** double-click a body to select it (the floor
  deselects), Shift+drag to pull it.
- **Driving:** the **Teleop** panel drives the newest or double-clicked
  robot: WASD walk, QE turn, or drag the goal marker on the floor; a
  gamepad works too. Robot arms have IK markers to drag.
- **Toolbar** (top middle): Pause, Reset (every robot back to its spawn pose),
  Clear (remove everything), Enter VR / AR where the browser supports it.
- **Panels**, left: **Assets** (the model library), **Stage** (what is
  spawned: poses, joints; save, load, download and share stages),
  **Physics**; right: **Teleop**, **Cameras** (the robots' cameras),
  **Render** (quality vs. GPU load, floor, lighting, camera follow),
  **Share** (open the page on another device).
- **Touch screens:** one finger orbits, two pinch and pan; double tap
  selects, press and hold then drag pulls.

## Local Development

In the browser alone (the physics in the page):

```sh
npm install      # postinstall copies the wasm/js runtimes into public/vendor/
npm start        # python3 scripts/serve.py 4007: public/ with Cache-Control: no-store
```

Open http://localhost:4007 (in JupyterHub: `.../user/<name>/proxy/4007/` — all
URLs are relative, so the path proxy works).

A world shared by every browser that opens it (the physics in Python):

```sh
pip install mujoco numpy aiohttp onnxruntime
npm run server   # python3 server/server.py 4007
```

Open http://localhost:4007. `?local` in the URL keeps the page local even
there.

The model library is `mjcf_assets/` next to `package.json`
(`<category>/<source>/<model>/<model>.xml`); `npm install` indexes it, and
`node scripts/vendor.mjs` again after adding models. Walking policies go in
`pretrained_policies/` ([Policies](docs/policies.md)).

## Docs

- [Spawning](docs/spawning.md) — the asset library, thumbnails, how the
  world is rebuilt on every spawn, memory
- [Physics](docs/world.md) — the Physics panel: options, timestep,
  floor, lighting, sleep
- [Teleop](docs/teleop.md) — driving, the goal marker, gamepads, scripted goals
- [Policies](docs/policies.md) — GR00T (G1), Playground (Spot, Go2), Microduck
- [Mobile bases](docs/mobile-bases.md) and [IK chains](docs/ik.md)
- [ROS bridge](docs/ros.md) — joint states, TF, commands, cameras over rosbridge
- [Cameras](docs/cameras.md) — the robots' cameras
- [Rendering](docs/rendering.md) — the Render panel, textures, materials
- [VR and AR](docs/vr.md) — WebXR controllers and menu
- [Shared world](docs/server.md) — the Python server and network mode
- [GitHub Pages and Binder](docs/deploy.md) — the static site, Binder, the Share panel
