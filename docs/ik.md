# IK chains

Every robot under `mjcf_assets/robots/` gets an `IkController`
(`public/js/controllers/ik/ik.js`), on by default: IK chains whose targets are RViz-style
interactive markers (`public/js/app/marker.js`, a fixed 0.15 m in the world):
drag an arrow to move the target along that axis, a ring to turn it about
that axis, the center ball (colored by chain) to move it in the plane
facing the camera. Axes are the target's own (red x, green y, blue z); the
part under the pointer turns yellow, and the camera does not move while a
marker is held. **G** opens or closes the gripper of the chain dragged
last. With the pointer over a marker (no button needed; while dragging it
too), the **mouse wheel** closes its gripper (towards you) or opens it
(away), 5 % a notch (1 % with Shift), from where it is, with a label by the
pointer; the camera zooms only elsewhere. Until a marker moves, a chain just holds its joints where they are.

A chain is only `{root, tip}`, plus the tip frame: a site rigidly attached
to the tip body (a TCP) or the body moved by an offset (Franka's 0.1034 m):
its joints are the ones on the body path from the tip up to the root,
root's own joints included, so moving the root to the elbow or down to
Garmi's lift changes which joints take part. What moves a joint, in
order: its own position servo; a position servo on a fixed tendon through
it (Stretch 3's telescoping arm: one IK dof for the tendon's length, the
joints sharing the motion); else, with no actuator at all, a software
servo of IK's own (every physics step, a critically damped PD force on the
joint scaled by its inertia, plus gravity compensation, in
`qfrc_applied`). Joints another controller drives (G1's waist and legs
under GR00T), free joints and joints the user unticks stay still; a joint
coupled to another by a joint equality (Garmi's second lift stage,
Stretch's telescope) adds its motion to its leader's Jacobian column. On a
robot with no position servos at all, automatic chains also take its
joints that have a range.

Configs come from, in order: the one saved in the panel, `IK_DEFAULTS`
(Garmi: both arms, root `*_fr3_link1`, tip `*_fr3_hand` + 0.1034 m), or
`autoConfig()`, which looks at the model: joints with a position servo,
not driven by another controller, not on the robot's root body (a mobile
base) and not named like hand/gripper joints (`finger`, `jaw`, `thumb`,
`hand_`, …) are movable; every serial run of at least 3 of them that ends a
branch of the body tree becomes a chain, from where the tree branches (or
the base) to the last movable body, carried on through fixed single
children (flange to hand). Shorter end runs (a pan-tilt head, fingers) are
dropped first. The tip frame is the model's TCP site when one is fixed to
the tip, by name, best first: `tcp`, `pinch`, `grasp`, `gripper` /
`end_effector` / `ee`, `hand_center` / `palm`, then the `attachment_site`
flange (i2rt YAM `tcp_site`, Kinova `pinch_site`, ViperX `pinch`, ALOHA
`*/gripper`, SO-101 `gripperframe`, TidyBot `pinch_site`, UR5e / iiwa /
Lite6 `attachment_site`). MJCF has no TCP element, so a model without such
a site (menagerie's `panda.xml`, G1) keeps the body origin;
`IK_DEFAULTS` gives Panda and Garmi the Franka TCP offset instead.

A chain's **gripper** is a list of actuators, each with the ctrl that opens
it and the one that closes it (`{actuator, open, closed}`; an older
config's single actuator name opens at the top of its ctrlrange, as the
Franka hand). Found automatically: every named actuator on joints below the
tip that the chain does not have and no other controller drives (a lone
one on a single hinge only when named like a gripper joint or followed by
another joint: else it is likely a wrist). Which end of its range closes is
worked out from the model: with its joints at that end, the geoms below the
tip lie closer together (fingers meet, or curl into the palm). So TIAGo's
two finger servos, Talos's gripper and G1's seven hand actuators per hand
are found as well as single-actuator grippers. G toggles
all of a gripper's actuators together (closed:
its first one nearer its closed ctrl); the panel's **Closed** slider, the
wheel over the marker and a VR trigger set it part way.
Examples: Panda `link1 → hand` (gripper
`actuator8`), UR5e `shoulder_link → wrist_3_link`, ALOHA and TIAGo dual
both arms, Apollo arms, legs and neck, Spot four legs, G1 both arms (legs
and waist are GR00T's), TIAGo `torso_lift_link → arm_7_link` (lift
included); Go2 and H1 have torque motors, so none.

Each tick (50 Hz) every chain takes one damped-least-squares step
(`dq = Jᵀ(JJᵀ+λ²I)⁻¹e`, Jacobian from `xaxis`/`xanchor`), with a pull
towards the starting posture in the null space, joint speed limits and
joint/ctrl range clamps, tracking position and orientation (orientation
weighted 0.3 m/rad; a chain with fewer than 6 joints gets the least-squares
compromise). A joint the step would push past its range is taken out and
the step solved again, so the rest do not chase what it cannot do. IK runs
on the *commanded* joint positions (forward kinematics in a scratch
`MjData`), so gravity sag does not wind the targets up. The target lives in
the frame of the parent of the first body the chain moves: it travels with
the base (and the lift, when the chain starts above it), and a chain rooted
at G1's pelvis keeps its target on the torso while it walks.

The Teleop panel edits the driven robot's chains: tip and root (select, or
double-click a body and press ◎), TCP site or offset, joints, and the
gripper (**+ actuator…** adds one, each with its open and closed ctrl, ⇄
swaps them, ✕ drops it, **Auto** finds them in the model again);
**+ Chain** starts one at the selected body. **Save** keeps the setup per
robot file in localStorage (`mujoco-filament/ik/<path>`), **Auto** /
**Defaults** goes back to the found / built-in one, **Export** copies it as
JSON for `IK_DEFAULTS`. While IK is on it owns its joints' controls, so the
Stage panel's sliders for them are overridden; untick **IK chains** to set
them by hand. Known limits: a straight, singular arm (RB-Y1 at rest) cannot
move along itself until moved another way first; joints driven through a
tendon (Stretch's telescoping arm) are not IK joints.
