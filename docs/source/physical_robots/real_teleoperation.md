# Physical Robot Teleoperation

Drive real arms live with HandUMI, and optionally record what they do as a
joint-level dataset. This is the entry point for everything that commands
hardware; the robot-free capture flow is
[Record Demonstrations](../record.md).

Validate the same robot in
[Simulated Teleoperation](../teleoperation.md) first. Tracking, calibration,
and motion mapping must already behave correctly there — the control flow is
identical and only the destination changes.

## Support by Robot

The HandUMI tracking and control flow remains the same for every robot.
Physical teleoperation additionally requires a backend for the selected
manufacturer and model; simulation or replay support alone does not imply
hardware support.

| Robot | Live simulation | Real teleoperation |
| --- | --- | --- |
| Piper | Supported | Supported |
| OpenArm v1 | Supported (kinematic) | Supported through optional `openarm` backend |
| TRLC-DK1 | Supported (kinematic) | Not yet supported |
| Axol | Supported | Not yet supported |
| Other robots | Add an embodiment | Add a hardware backend |

A robot with real teleoperation support also supports `handumi replay-real`,
which drives the same backend from a converted dataset.

See [Add a New Robot Embodiment](../development/new_embodiment.md) for the
common interface used to add future manufacturers and models without changing
the HandUMI capture workflow.

## Hardware Preparation

Complete the hardware-specific preparation before commanding a physical robot.
Both guides start with single-arm validation before enabling both arms:

- [Piper Hardware Setup](piper_setup.md)
- [OpenArm v1 Hardware Setup](openarm_v1_setup.md)

## Teleoperate

The general interface is:

```bash
handumi setup --robot <robot_id> --device meta
handumi teleop-real --robot <robot_id> --device meta
```

The input/IK loop and robot command clock are independent. Real teleoperation
uses the incremental DLS follower at 72 Hz by default and a delayed trajectory
stream sends interpolated commands at 100 Hz. The delay automatically follows
the requested input rate (one input frame plus a small scheduling margin), so
the 72 Hz default uses about 20.6 ms instead of the historical 40 ms at 30 Hz.
`--fps` controls how often HandUMI polls and processes the latest headset pose;
it does not change the headset application's own render refresh rate. Use
`--ik-solver lm` to compare the legacy solver, which retains a 30 Hz default,
or override either mode explicitly with `--fps <hz>`. The DLS joint clamp is
time-based, so changing this rate does not change maximum joint speed.
Its per-joint speed is resolved automatically as the minimum of the joint's
URDF velocity limit, the real backend's configured limit, and HandUMI's
conservative 1 rad/s teleoperation ceiling; it does not require a duplicate
`ik_weights.max_joint_speed_rad_s` robot setting.

`--trajectory-delay-ms` can still override the automatic delay and controls the
smoothness/latency tradeoff. Inspect all shared advanced options with
`handumi teleop-real --help-advanced`.

Camera previews use the `left_wrist`, `right_wrist`, and `workspace` entries
in `configs/rig.yaml`. Any of them may use `type: zedmini`; it keeps the same
logical name and displays only its left `672×376` image. See
[Setup and Calibration](../setup.md) for the rig schema.

## Record a Real-Robot Dataset

Use `handumi teleop-record` when the real robot should be driven live and saved
as a joint-level dataset:

```bash
handumi teleop-record --robot <robot_id> --device meta \
  --output-dir outputs/my-dataset
```

For an unlimited Piper/PICO collection such as Tower of Hanoi:

```bash
uv run handumi teleop-record \
  --device pico \
  --robot piper \
  --side both \
  --num-episodes 0 \
  --task "Build the Tower of Hanoi" \
  --output-dir outputs/hanoi
```

`--num-episodes 0` keeps the session available until the operator finishes it.
Use a new output directory, or add `--resume` when appending to a finalized
dataset in the same directory.

This command has its own parser and operational defaults. It does not use
`--record` on `handumi teleop`; the plain `handumi teleop` command is reserved
for live simulation. `--resume` with the same `--output-dir` verifies and
appends to that finalized local dataset.

`teleop-record` uses the same 30 Hz IK → delayed 100 Hz command trajectory and
the same camera backends as `teleop-real`. The selected camera streams are
stored as native LeRobot v3 video features alongside the joint-level state and
action columns. Each MP4 is written under
`videos/observation.images.<camera>/chunk-*/file-*.mp4` and is referenced by
the episode metadata in `meta/episodes/`.

Camera acquisition, Rerun preview, robot command playback, video encoding, and
LeRobot dataset writing run independently. The 30 Hz control loop only queues
an aligned dataset row; it does not wait for MP4 encoding. The writer queue is
bounded so an encoder that cannot keep up causes the current episode to be
discarded instead of increasing robot-control lag or silently dropping frames.

### Episode gestures

The two grippers control continuous episode collection:

1. The robot starts at home. Double-squeeze the **right HandUMI gripper** to
   start recording. The gesture itself does not activate an arm; reopening a
   HandUMI gripper wakes and anchors its corresponding parked arm.
2. While recording, double-squeeze the **right gripper** again to save the
   episode. The command waits for the real backend to finish returning the
   enabled arms home, then enters `READY` without starting another episode.
3. Double-squeeze the **left gripper** to discard the active episode. The robot
   likewise returns home and waits in `READY`.
4. Reset the physical task while the robot is at home, then double-squeeze the
   right gripper to start again.
5. Double-squeeze **both grippers** to discard the active episode and finish
   the session. `Esc` and `Ctrl+C` also discard and stop.

The bilateral gesture may be staggered by up to 200 ms. This prevents small
sampling differences between grippers from turning a discard into a start or
save. `Esc` and `Ctrl+C` discard the active episode and stop the session.

With `--space-start`, Space remains an alternative way to start an episode
from home. When Feetech grippers are connected it does not activate the arms;
opening each HandUMI gripper still does that. With `--skip-feetech`, Space is
also the explicit arm-start fallback.

### Recording display

`teleop-record` uses the same full-screen terminal dashboard as `handumi
record`. It shows the task, dataset, episode, wall-clock and effective data
times, frame totals, live gripper widths, and an operator guide. Its state
changes through `READY`, `RECORDING`, and `HOMING`; do not reset the physical
task until it returns to `READY`. While recording, control timing and writer
queue diagnostics are updated without blocking robot output. A growing writer
queue indicates storage or encoding pressure; the episode is discarded if the
bounded queue fills.

Holding a physical **HandUMI** gripper completely closed for the configured
hold period (two seconds by default) parks only that arm at home. Keeping it
open cancels the timer. Opening that same HandUMI gripper wakes and re-anchors
the parked arm.
Double-squeeze episode gestures continue to use the HandUMI grippers.

Unlike a robot-free HandUMI capture, this dataset is bound to the robot that
produced it. To collect demonstrations that can be retargeted to any supported
embodiment later, record with HandUMI alone.

## Replay a Dataset on the Real Robot

`handumi replay-real` plays a converted joint-level dataset on the physical
robot with no IK and no tracking device: the stored joints are streamed
through the same backend, at the recorded rate, through the same interpolated
100 Hz command stream as `teleop-real`. Use it to prove a dataset is
deployable before training on it.

```bash
handumi replay-real outputs/datasets/my-dataset-bi_piper_follower \
  --robot piper --episode 0 --dry-run
handumi replay-real murobotics/my-dataset-bi_piper_follower \
  --robot piper --episode 0 --speed 0.5 --side left
```

A Hugging Face repository id is fetched on first use and cached under
`outputs/datasets/<repo-name>/`, the same local root a path argument would
use. The full repository tree (`data/`, `meta/`, `videos/`) is downloaded.

`--robot` selects the backend exactly as in `teleop-real`; it defaults to the
robot the dataset was converted for and refuses a dataset converted for
another one. `--rig-config`, `--skip-can-repair` and `--home-pose` mean what
they mean in `teleop-real`; `--state-layout` and `--use-degrees` mean what
they mean in `replay-joints` and are normally read from the dataset.
The dry run decodes the episodes, checks joint limits and joint speed,
predicts the command stream's own tracking error and prints the table
placement the dataset was converted for, without opening any CAN connection.
Frame selection and stream timing are under `--help-advanced`. The hardware
run:

1. asks for confirmation (`--yes` skips it), homes at the slow homing speed,
   and ramps into the first frame over `--approach-seconds` (default 3 s);
2. streams the frames, slowed by `--speed`, while logging measured joints and
   gripper openings; arms outside `--side` stay parked at home;
3. returns home slowly, or stays on the last frame with `--no-return-home`.

Ctrl+C or a backend fault holds the arms where they are and disables them; no
return-home motion is attempted. Each episode prints the tracking lag, the
joint and TCP error after lag compensation, and a PASS/FAIL verdict against
`--tolerance-deg` / `--tolerance-mm`; `--strict` makes a FAIL exit non-zero.
Logs go to `outputs/replay_in_real/<dataset>/episode_*.npz` and `.json`.

### What real replay adds over simulation

`handumi replay-joints` and `replay-real` send the same joint values, so
limits, gripper mapping, reachability and the TCP path are already proven by
the simulation replay. The hardware run measures only what simulation cannot:

- **Speed.** HandUMI captures the human's own pace, and conversion keeps it
  (up to 0.35 rad per frame). `teleop-real` caps joints at 1 rad/s and the
  operator slows down to match, so converted data is faster than any teleop
  recording; a demonstration can even exceed the arm's own limit (the tblock
  Piper set peaks above 180 deg/s in most episodes). The simulation is
  kinematic and follows anything; the real backend obeys `real.*` in the
  robot YAML. Piper's command stream plans an acceleration-limited move
  toward the latest target, and that braking envelope lags a moving target
  by `v^2 / 2a`: with the teleop value of 720 deg/s^2 that is 7 deg at
  100 deg/s and 22 deg at 180 deg/s, invisible under the teleop cap but not
  at demonstration speed. The dry run plays every episode through the same
  generator and prints the predicted lag and error, so a run that cannot pass
  is known before any connection opens. Start at `--speed 0.5`; raise
  `--accel` (2880 took the tblock prediction at half
  speed from 5 to 10 deg down to 1 to 3 deg) rather than lowering the
  tolerance. OpenArm's stream is a plain rate limiter and ignores that flag.
- **Table placement.** The dataset was retargeted for the table pose in its
  `handumi.deployment_calibration`, and that pose is baked into the stored
  joints: replay cannot move it, so the plan prints it (the sim profile is
  centered between the bases at base height, 0.45 m forward for Piper) and
  flags an unverified or simulation-only profile. The physical bases must
  match the URDF separation (0.60 m for the bi-Piper) and the table surface
  must sit at base height, or the fingertip will land at the wrong height.
  Check the lowest TCP height of an episode in the screening report
  (`meta/handumi_screening_<robot>.md`, `table` column) before replaying it,
  and clear the table for the first run. Objects go where the demonstration
  put them: for a tracking check they are not needed.
- **No filter.** Teleop smooths IK output with a One Euro filter; replay
  sends the stored joints unfiltered, interpolated at 100 Hz, because that is
  what a trained policy will emit and what the simulation showed. The
  backend's acceleration limit is the only low-pass. Add smoothing only if
  the arm visibly vibrates.
- **Verdict.** The joint tolerance compares the measured joints with the
  command after estimating the tracking lag (stream delay plus motor
  response; the report prints it). A FAIL caused by a few fast frames at
  full speed is a fact about the robot's speed limit, not about the data;
  a FAIL at `--speed 0.5` with the table clear points at the dataset or the
  installation.

## See the Cameras in the Headset

While teleoperating, the context and wrist camera feeds can be streamed into a
PICO headset so the operator does not have to look at the workstation screen.
It works independently of the selected robot and does not change the
teleoperation commands above.

- [PICO Remote Vision](pico_remote_vision.md)

## Safety

Keep the workspace clear and an emergency stop accessible. Enforce joint,
velocity, acceleration, workspace, and collision limits. Run
`handumi teleop-real --help` for backend-specific options.

To inspect an existing recording rather than live motion, continue with
[Replay a Local Recording in Simulation](../workflows/replay_in_sim.md), then
run the checks in [Quality Assurance](../workflows/datasets.md).

```{toctree}
:hidden:
:maxdepth: 1

pico_remote_vision
```
