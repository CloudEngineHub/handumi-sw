# Laboratory-local table calibrations

Copy `example.yaml` to `<robot>.yaml`, then set the robot, laboratory ID, and
measured `robot_from_table` pose. For example:

```bash
cp configs/calibration/table/local/example.yaml \
  configs/calibration/table/local/piper.yaml
```

Real `*.yaml` files in this directory are intentionally ignored by Git. They
describe one physical installation and must not be published as universal
robot configuration. `example.yaml` is the only versioned YAML here.

Set the same stable laboratory identifier under `deployment.lab` in the
machine-local `configs/rig.yaml`. Replay discovers
`configs/calibration/table/local/<robot>.yaml` automatically. Use
`deployment.table_calibrations.<robot>` only to override that conventional
path, for example when a laboratory stores private calibration elsewhere.
