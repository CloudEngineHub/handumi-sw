# Robot/table deployment calibrations

These transforms place the robot-agnostic HandUMI table frame in a target
robot's world. They are target deployment configuration, not part of the raw
recording and not Controller-to-TCP offsets.

```text
table/
├── sim/                 committed canonical simulation layouts
│   └── <robot>.yaml
└── local/               one physical laboratory's private measurements
    ├── example.yaml     committed template
    └── <robot>.yaml     ignored by Git
```

Use `--deployment-profile sim` for reproducible public simulation results and
`--deployment-profile local` when validating a measured physical installation.
The default `auto` profile selects a conventional local file when it exists and
otherwise falls back to the canonical simulation file. Replay always prints
the resolved profile and path.

Each local calibration must use `scope: physical`, declare the same stable
`lab` identifier configured under `deployment.lab` in `configs/rig.yaml`, and
remain `verified: false` until the hardware checks pass. A laboratory may use
`deployment.table_calibrations.<robot>` in `configs/rig.yaml` to override the
conventional local path.

Dataset-specific placements (for example a scene recorded with the board away
from the demonstrated workspace) are visualization artifacts rather than
canonical configuration. Keep them under `outputs/calibration/table/` and pass
them explicitly with `--deployment-calibration`.
