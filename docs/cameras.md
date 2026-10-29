# Cameras panel

The **Cameras** panel (`public/js/cameras/cameras.js`) works like the Teleop
panel's IK chains: it shows one robot at a time (picked at the top; a
newly spawned one comes up) and its cameras, each in a foldable box. They
start as the MJCF `<camera>`s of the robot's model (24 of the library's
robots have some: wrist cameras on ALOHA, SO-101, Kinova, head cameras on
TIAGo and Microduck, Stretch 3's five), and every one of them can be
edited, deleted or added to (**+ Camera**: on the selected, double-clicked,
body if it is the robot's, else its base). A camera rides a **link** (**◎**
takes the selected body; "robot frame" the robot's own) at an **offset** in
the link's frame, **turned** by roll / pitch / yaw (degrees about the link's
x, y, z) from looking along the link's +x with +z up, so pitch > 0 looks
down; **follow** "position only" keeps its turn in the robot's frame while
following where the link is (how MJCF track / trackcom cameras come in).
Each has a size (width × height, 256×256 unless the model gives a
`resolution`), rate (5 to 30 Hz) and vertical field of view.

Ticking a camera opens it in a picture-in-picture window (drag it by its
title bar; its **◉** looks through it in the main view, Esc: back); **⌖**
shows where it is in the scene: a frustum of its field of view and aspect,
a green tick on its top edge. **Save** keeps the robot's cameras for every
spawn of its model in this browser; without, a model starts from its
built-in setup in `public/js/cameras/defaults.json` (keyed like
`controllers/ik/defaults.json`: the file under `mjcf_assets/robots/`, other models by
theirs under `mjcf_assets/`), else the model's cameras. **Export** copies
the setup as JSON for that file, **Defaults** / **Auto** goes back to it /
to the model's.

Local and network mode alike: the pose comes from the world's body poses
(MuJoCo's in local mode, the page's own forward kinematics in network
mode); an MJCF camera comes in at the pose MuJoCo gives it (`cam_xpos` /
`cam_xmat`), its definition from the robot's model (the server's render
data has it in network mode).

Each camera is a Filament View rendered after the window's frame into a
framebuffer of its own: the page makes Filament bind it where it binds the
default framebuffer (as for WebXR), so the main canvas is untouched. The
pixels are read back asynchronously (into a pixel buffer, collected once a
fence has passed), so the window keeps its frame rate: on an RTX 3080 a
320×240 camera at 30 Hz costs the main loop nothing visible. The viewer's
own markers (selection, IK and goal markers, origin axes, VR controllers
and menu) are on a layer the cameras do not see; their views take the
Render panel's anti-aliasing, SSAO, reflections and shadows, like the window
(more GPU time per camera frame at the higher settings).
