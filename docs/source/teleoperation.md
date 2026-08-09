# Teleoperation in Simulation

HandUMI produces robot-agnostic live tool poses and gripper commands. A selected
robot embodiment maps those commands to its kinematics. This page drives that
model in simulation only: nothing is sent to hardware and nothing is recorded.

Validate tracking, calibration, and motion mapping here first, then continue
with [Physical Robot Teleoperation](physical_robots/real_teleoperation.md).

## Live Simulation

Select any supported robot model through `--robot`:

```bash
handumi teleop --device meta --robot <robot_id>
```

For example, using the currently supported Piper embodiment:

```bash
TARGET_ROBOT=piper
handumi teleop --device meta --robot "$TARGET_ROBOT"
```

OpenArm v1 uses the same command and starts from its configured `home_q`:

```bash
handumi teleop --device meta --robot openarmv1
```

This opens Viser with the live robot model and Rerun with tracking, TCP trails,
gripper widths, and both wrist cameras. Nothing is recorded. Use `--device pico`
for PICO.

Add a task scene with:

```bash
handumi teleop --device meta --robot "$TARGET_ROBOT" --scene cube_in_box
```

Teleoperation shows both wrist cameras by default, so `--cameras` is only
needed to change that selection — for instance to add the overhead view:

```bash
handumi teleop --device meta --robot "$TARGET_ROBOT" \
  --cameras left_wrist,right_wrist,workspace
```

It accepts the logical names `left_wrist`, `right_wrist`, and `workspace`;
their physical device IDs come only from the corresponding entries in
`configs/rig.yaml`, which is where each camera is declared once. Use
`--skip-cameras` to run without any camera view.

Camera backend, capture resolution, output resolution, and FPS come from each
logical view's `cameras.<name>` entry. If one of the three views uses
`type: zedmini`, it is shown as its cropped `672×376` left image without
changing the `--cameras` logical name.

Viser shows the robot and Rerun shows tracking and camera trails. Use `--no-rerun` or `--no-viser` when a viewer is not needed.

### Motion timing

Tracking and IK run at `--fps` (30 Hz by default). Their timestamped joint
targets enter a short delayed buffer; an independent stream samples that
trajectory at `--command-rate-hz` (100 Hz by default) using
`t - --trajectory-delay-ms`. This removes 30 Hz command steps without running
IK repeatedly on the same tracker pose. Simulation, real teleoperation, and
teleop recording use the same pipeline. Advanced tuning is available through
`handumi teleop --help-advanced`.

### Start and Reset

Arms sit idle at home until they are started, and the same gesture stops them
again:

- **Double-squeeze a gripper**: start the enabled, tracked arms from home.
- **Double-squeeze again**: clear the anchors and return them home. This is the
  stop.
- Tracking loss cancels pending motion and holds the latest command.

Two optional ways to start exist for when squeezing a gripper is impractical:

- `--space-start`: also start idle arms by pressing Space in the terminal.
  Space only *starts*; it is not a stop or pause key.
- `--auto-start`: start on their own once controller tracking has been valid
  for `--auto-start-delay-s` (default 5), with no gesture at all.

`handumi teleop-record` uses side-specific gestures for episode collection:
left starts, right saves and advances to an automatically started next episode,
and both grippers together discard. See
[Record a Real-Robot Dataset](physical_robots/real_teleoperation.md#episode-gestures).
