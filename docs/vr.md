# VR and AR

**Enter VR** / **Enter AR** are in the toolbar; what the headset renders
is in the [Render panel](rendering.md) notes.

**Controllers** (`public/js/render/xr.js`; Quest Touch, or any xr-standard
gamepad), in local and network mode. Each controller points a ray, which
ends on the menu, the IK marker or the floor it meets:

- **Grip** on an IK marker of the driven robot, reached by hand or with the
  ray: the target follows the controller while held, moved as the hand moves
  and turned as it turns (about the marker's center). Let go and grab again
  to go on, like a clutch; each hand can hold its own chain (two arms at
  once).
- **A** (right controller), held on any body under the ray: pulls it with
  the selection spring, like Shift+drag on the desktop, held at the end of
  the ray as on a stick. (The grip did this too at first, but just holding
  the controllers squeezes it.)
- **Y** (left controller), held 0.4 s, opens the **menu** (and closes it;
  a short press closes it too): a panel in front of you, the trigger
  choosing the row under the ray (`public/js/render/xrmenu.js`, menus of submenus
  with a Back row). **Reset scene** puts every robot back to its spawn pose
  (as the toolbar's Reset; in a shared world for everyone); **Quality** sets the headset's **Resolution** (0.6× to 1.5×
  its default framebuffer, at once), **Anti-alias** (MSAA off, 2×, 4×),
  **Shadows** (the sun's, off by default), **Brightness** (the eyes'
  exposure, 100 % down to 25 % of the desktop's: the headset shows the scene
  brighter than a monitor; these four are kept in this browser) and the
  scene's **Textures** (as the Render panel); **FPS**
  shows the session's frame rate.
- **Trigger** while the same hand holds a marker: closes that arm's gripper
  as far as it is pressed (released: open), once pressed at all during the
  hold. On a marker under the ray: opens or closes its gripper. Held on the goal marker: drags it over the floor like the
  mouse, the same controller's stick sideways turning it; the goal is set
  on letting go. On a robot: selects it and drives it (like a
  double-click); on anything else, releases it.
- **Left stick** walks the driven robot (forward/back, turn) while pushed,
  whatever the Teleop panel's input; it takes over from the robot's goal,
  which is dropped.
- **Right stick** sideways turns you by 30° about where you stand; pushed
  forward it aims at the floor (the spawn marker shows where), and letting
  go puts you there. In AR this moves the scene in your room.

**Hand tracking** (Quest 3 and others with WebXR Hand Input; the session
asks for `hand-tracking`, optionally): put the controllers down and your
hands take their place (a hand still holding a controller is that
controller: its gestures are not read), each pointing the same kind of ray, its skeleton
(25 joints and the bones between them) drawn in its color. The gestures stand for the buttons above:

- **Pinch** (thumb and index tips together) is the **trigger**, as far as
  they are close: rows of the menu, the gripper of a marker under the ray,
  the goal marker, selecting a robot.
- **Palm on a marker**: hold the middle of your palm (between the wrist
  and the middle finger's knuckle) within 5 cm of an IK marker's center
  for 3 s (it lights up, and a countdown by your hand shows the seconds left)
  and it sticks to your palm, following
  it as a held grip does. The robot's fingers then follow yours: an
  actuator whose gripper entry names a `source` (an angle of your hand: a
  knuckle's bend, a finger's sideways spread, the thumb's base, …; [IK](ik.md),
  `public/js/render/handmap.js`) as far as that angle is between your open
  hand and your fist, the others by their name (thumb, index, middle, ring, pinky /
  little), closing as far as your finger is curled (the angles between its
  bones added up; one on no named finger as the hand is on average). A
  gripper with no named fingers (Panda's) closes as far as you pinch. The
  menu's **Calibrate hands** measures your hands: Start, then both hands
  wide open (4 s), then fists (4 s); each angle's value open and as a fist
  is kept in this browser (Back to a typical hand forgets it). Pinches and
  the menu gesture do nothing meanwhile. The menu's **Hand angles** shows
  each hand's angles live (degrees). If the headset loses track of the hand, the
  marker stays stuck where it is: the hand coming back within 8 cm of where it
  went and within 2 s takes it up again from there; further away or later,
  the marker is let go (back onto the tip).
- **Letting go**: opening the menu (the left pinch below) or pressing the Meta button (the
  system's menu: the session loses focus) lets go of every marker the hands are stuck to.
  While a hand is stuck, the menu is the only other gesture (the other hand can still stick
  to another marker by holding its palm on it).
- **Right hand pointing** (index straight, middle, ring and little fingers
  curled in) aims a **teleport**: the ray's spot on the floor shows where
  (the spawn marker). Open the hand all the way and you are there; curl it
  into a full fist instead to call it off.
- **Left pinch with the palm towards you**, held 0.4 s, opens or closes the
  **menu** (Y); opening it lets go of the stuck markers.
- Otherwise there are no sticks and no pulling bodies: walking the robot,
  snap turns and pulling stay on the controllers.
- **Modes** (the menu's **Mode**, kept in this browser):
  - **Teleop** (the default): the gestures and buttons above drive the robots; your hands
    collide with nothing.
  - **Avatar**: you are in the physics as two solid hands and a head (seen, colliding with nothing)
    (`public/js/world/avatar.js`: a ball per hand joint, a controller its wrist; mocap bodies
    after every robot, not under gravity), so you push, tip over and poke objects and robots.
    Mocap bodies are not pushed back: the hands go through what they cannot move (a fast hand
    can pass through a small object) and hold nothing. The hands only push here: no sticking
    to markers, no grabbing or pulling, the trigger only on the menu, the left stick no longer
    walks the robot; the menu, teleporting and turning stay. In a shared world everyone sees
    your avatar in your color (`public/js/render/avatars.js`; the server has 4 avatar slots,
    `server/avatar.py`, one per client in Avatar mode, free again when it leaves the mode or the
    world). Rays pass through avatars.
