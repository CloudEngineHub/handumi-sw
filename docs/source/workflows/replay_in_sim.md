# Replay a Local Recording in Simulation

`handumi replay` retargets one raw HandUMI episode to a configured
bimanual robot and displays the target and achieved TCP trajectories in Viser.
The source recording remains robot agnostic: the robot model, IK profile, and
table placement are selected when replay starts.

## Install Replay Dependencies

Install the Python simulation extra before the first replay:

```bash
uv sync --extra sim
```

On a PICO workstation, use `bash install.sh --sim` in place of the final
`uv sync` command. This preserves the locally built XRoboToolkit package by
reinstalling it after the dependency sync. XRoboToolkit is not needed when the
workstation only replays existing datasets.

Trajectory replay reads Parquet state columns directly and does not decode the
dataset's MP4 camera features. FFmpeg is therefore not required for IK replay.
Install and verify it when inspecting or curating camera videos:

```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -version
ffprobe -version
uv run python -c "from torchcodec.decoders import VideoDecoder; print('TorchCodec OK')"
```

If this reports `Could not load libtorchcodec`, see
[Dataset Video Loading Fails in TorchCodec](../troubleshooting.md#dataset-video-loading-fails-in-torchcodec).

## Use a Local Dataset

Pass the recording directory as `DATASET`. A local replay does not download
data:

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/20260714_224135 \
  --robot openarmv1 \
  --episode 0
```

Change `--episode` to select another episode. Add `--headless` when only the
IK result and saved NPZ are needed. Without it, open the URL printed by Viser,
normally <http://localhost:8080>.

`JAX_PLATFORMS=cpu` is recommended on workstations that have the JAX CUDA
plugin but not a working CUPTI installation. Without it, JAX can print an
`Unable to load cuPTI` traceback and then continue on CPU; the message does not
mean that replay or IK failed.

## Absolute-table Retargeting

Recordings captured in the calibrated table workspace normally select
`absolute-table` automatically. The explicit form is useful when auditing a
new embodiment:

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/20260714_224135 \
  --robot openarmv1 \
  --episode 0 \
  --retarget-mode absolute-table \
  --deployment-profile sim
```

`robot_from_table` places the demonstrated table frame in the robot world. It
does **not** move the robot base. For OpenArm v1, for example, the URDF pedestal
remains fixed to world `Z=0`, the shoulder mounts are at `Z=0.698 m`, and the
simulation calibration's `Z=0.28755 m` is the provisional table-plane height.

The deployment profiles intentionally have different ownership:

| Profile | Location | Meaning |
| --- | --- | --- |
| `sim` | `configs/calibration/table/sim/<robot>.yaml` | Portable canonical simulation layout; committed, `scope: simulation`, never a claim about a lab |
| `local` | ignored `configs/calibration/table/local/<robot>.yaml` | Measured placement of one laboratory's physical robot and table; `scope: physical` |
| `auto` | local when configured, otherwise sim | Convenient default; the resolved profile and file are always printed and saved |

An explicit `--deployment-calibration FILE` overrides all profiles. Do not edit
the canonical simulation file to match a physical lab and do not use
`robot_from_table` to compensate for an incorrect Controller-to-TCP
calibration.

### Configure a Laboratory Placement

Create one private calibration per installed robot. The destination is ignored
by Git:

```bash
cp configs/calibration/table/local/example.yaml \
  configs/calibration/table/local/piper.yaml
```

Set the copied file's `lab` to the same stable identifier used below, measure
`T_robot_world_table`, and keep `verified: false` until the physical touch
checks pass. Select it in `configs/rig.yaml`:

```yaml
deployment:
  lab: my_research_lab
```

Replay discovers `local/piper.yaml` automatically. Use the optional
`deployment.table_calibrations.piper` setting only when the lab stores its
private file outside this conventional directory.

Then require it explicitly during a lab check:

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/datasets/tblock \
  --robot piper \
  --episode 0 \
  --deployment-profile local
```

`handumi calibrate verify --robot piper --device pico` rejects a simulation
file as a physical calibration and reports an unverified local file.

### Use a Dataset-Specific Placement for Visualization

The canonical placement assumes the session board sat at the near edge of the
task scene, so the demonstrations happen on the table's `+Y` side. When a
recording placed the board beyond the task zone instead (the demos then sit at
negative table `Y`), the mapped targets land on the robot bases and the strict
start check fails. Diagnose this with the printed
`source TCP workspace bounds` line: a mostly negative `Y` range means the
board placement, not the calibration files, is the problem.

For such datasets, write a dataset-specific table YAML that shifts the scene
back in front of the arms and pass it explicitly. For `tblock` with Piper:

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/datasets/tblock \
  --robot piper \
  --episode 0 \
  --retarget-mode absolute-table \
  --deployment-calibration outputs/calibration/table/tblock_piper_viz.yaml \
  --initial-position-tolerance-m 0.05
```

This preserves the demonstrated world-space motion for inspection. It is not a
fidelity claim: for `tblock` episode 0 the median tracking error is `2.4 cm`,
but the final segment still exceeds `10 cm` because the demonstration leaves
the BiPiper's bimanual workspace under any rigid placement. Keep such YAMLs
under `outputs/calibration/table/` rather than the canonical `sim/` directory,
and prevent the root cause during collection by placing the ChArUco board so
the manipulation happens in front of it (`+Y`, away from the operator).

Deriving the correction is pure post-processing; the raw dataset never needs
to change. Frame reminder: in the table frame `+Z` is up and `+Y` is the
horizontal depth away from the operator (unlike the headset world, where `+Y`
is up), so a mostly negative `Y` range is a horizontal offset, not a height
problem. Read the demo volume from the printed
`source TCP workspace bounds`, then shift the canonical placement's position
until that volume sits centered in front of the arms, leaving the quaternion
untouched. For `tblock`, bounds of `y = [-0.273, 0.045]` (centered near
`-0.11 m`) turned the canonical `[0.30, 0.0, 0.0]` into
`[0.5422, -0.1106, 0.0446]`: forward by roughly the demos' depth behind the
origin, and re-centered laterally. The same recorded episodes replay unchanged
on any robot once that robot's own placement is adjusted the same way.

## OpenArm v1

The current OpenArm profile uses a larger offline-only joint step than live
teleoperation:

```yaml
replay:
  max_joint_delta: 0.35
```

This does not change the real OpenArm command rate, speed limits, watchdog, or
following-error checks. The simulation URDF also keeps approximately `0.48 mm`
of clearance between the finger collision meshes at the closed `0 mm`
position. The real backend retains its native closed/open motor calibration.

For `outputs/20260714_224135`, the provisional rigid table transform produces:

| Episode | Maximum TCP position error | Result against 3 cm threshold |
| --- | ---: | --- |
| 0 | 2.92 cm | Pass |
| 1 | 4.71 cm | Review unreachable segment |
| 2 | 4.34 cm | Review unreachable segment |

Do not simply reduce the table translation in X to hide those peaks. In the
same recording, values at or below `X=0.168 m` make episode 0 cross a singular
branch and create 17--20 cm errors. A future reach limiter or workspace scaling
policy is preferable to distorting the measured table transform.

## TRLC-DK1

TRLC-DK1 currently supports bimanual kinematic replay in simulation. It does
not yet provide a HandUMI real-hardware backend.

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/20260714_224135 \
  --robot trlc_dk1 \
  --episode 0 \
  --retarget-mode absolute-table \
  --deployment-profile sim
```

The bimanual URDF uses two namespaced DK1 followers with a provisional `0.60 m`
base separation. The table transform is also provisional. On episode 0 of the
recording above, the current profile produced `0.22 cm` maximum position error
and `22.19 degrees` maximum orientation error.

TRLC meshes use paths such as `meshes/visual/base_link.glb`, resolved relative
to `assets/trlc-dk1`. If Viser prints `Can't find meshes/...` and shows only
trajectory lines, update the checkout and restart the replay process so the
URDF is loaded again.

## Axol

Axol supports bimanual kinematic replay in simulation with the same automatic
absolute-table flow:

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/datasets/handumi-screws \
  --robot axol \
  --episode 0
```

The Axol URDF uses `+X` toward its left arm, `+Y` toward the operator, and
`+Z` upward. Its provisional simulation calibration therefore rotates the
HandUMI table frame 180 degrees about Z and places the demonstrated workspace
at `[0.05714, 0.12376, 0.25022]` m in Axol world. This placement is fitted to
the complete three-episode validation recording and remains `verified: false`;
it is not a physical table measurement.

With the configured offline replay joint step, all three episodes pass the
default strict IK thresholds:

| Episode | Mean position error | Maximum position error | Maximum orientation error |
| --- | ---: | ---: | ---: |
| 0 | 0.04 cm | 2.72 cm | 9.30 degrees |
| 1 | 0.03 cm | 1.52 cm | 5.26 degrees |
| 2 | 0.03 cm | 0.38 cm | 7.05 degrees |

The supplied Axol model represents `left_gripper` and `right_gripper` as fixed
links. Recorded gripper openings remain in the rollout metadata, but the mesh
cannot visibly open or close until an Axol URDF with actuated finger joints is
available. Axol does not currently provide a real-hardware backend.

## Reading the Diagnostics

Replay prints the source tool identity and calibration hash before solving.
Seeing `source tool: robot=piper` while replaying OpenArm, TRLC, or Axol is
expected when Piper was the physical tool used to make the recording. The
identity-bound Controller-to-TCP snapshot reconstructs the demonstrated Piper
TCP; the target embodiment is applied afterward.

Important output fields are:

- `deployment calibration`: resolved profile, scope, verification state, and
  source file;
- `source TCP workspace bounds`: robot-agnostic table-frame volume used by the
  reachability check;
- `start prepared`: initial solve iterations and first-frame error;
- `IK EE error`: mean and maximum position/orientation error over both arms;
- `max_joint_delta`: the offline joint-step limit selected for the embodiment;
- `saved`: the NPZ containing targets, achieved TCP poses, errors, and qpos.

Use `--strict-ik` in automated validation. It exits when the maximum position
or orientation error exceeds the selected thresholds:

```bash
JAX_PLATFORMS=cpu uv run handumi replay \
  outputs/20260714_224135 \
  --robot trlc_dk1 \
  --episode 0 \
  --headless \
  --strict-ik
```
