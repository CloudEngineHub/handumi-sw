# HandUMI

Collect robot-free bimanual demonstrations once with HandUMI, then validate,
retarget, and reuse them across different bimanual arms with parallel grippers.

**Core Workflows** need no robot arm: record with HandUMI, preview an
embodiment in simulation, then replay and validate the result.
**Physical Robots** covers everything that commands real hardware.

```{image} _static/HandUMI.png
:alt: HandUMI hardware
:class: handumi-cover
:width: 100%
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

getting_started/installation
setup
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Core Workflows

record
teleoperation
workflows/replay_in_sim
workflows/datasets
workflows/dataset_curation
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Physical Robots

physical_robots/real_teleoperation
physical_robots/piper_setup
physical_robots/openarm_v1_setup
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Help

troubleshooting
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

development/new_embodiment
```
