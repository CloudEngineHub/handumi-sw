# Dexumi

Robotics project built on [LeRobot](https://github.com/huggingface/lerobot) with PICO retargeting, IK, and dataset tooling.

## Installation

```bash
./bin/install.sh
```

## Project layout

```
.
├── assets/                  # Robot URDFs and meshes (axol, piper, …)
├── bin/                     # Shell launchers
├── docs/                    # Architecture and guides
├── external_dependencies/   # XRoboToolkit bindings
├── outputs/datasets/        # Local LeRobot datasets
├── scripts/                 # Python tooling
├── src/dexumi/              # Core package
├── test/                    # Visualizers and dataset tests
└── utils/                   # Upload helpers
```

## Docs

| Doc | Description |
|-----|-------------|
| [docs/architecture.md](docs/architecture.md) | How the project is structured |
| [docs/add-new-embodiment.md](docs/add-new-embodiment.md) | How to add a new robot |

## Scripts

### `scripts/replay_pico_ik.py`

Replay a PICO dataset episode on a robot via IK (optional Viser visualization).

```bash
python scripts/replay_pico_ik.py --embodiment piper --episode 0 --visualize
```

| Argument | Description |
|----------|-------------|
| `--embodiment` | Robot: `piper` or `axol` |
| `--episode` | Episode index (default: `0`) |
| `--dataset-root` | Local dataset path (default: `outputs/datasets/<repo-id suffix>`) |
| `--workspace` | `rest` or `front` |
| `--visualize` | Open Viser scene with PICO skeleton + robot FK |
| `--save` | Save solved joints to `.npz` |

### `scripts/compare_axis.py`

Compare multiple PICO axis mappings side by side in a Viser grid.

```bash
python scripts/compare_axis.py --embodiment axol --episode 0
```

| Argument | Description |
|----------|-------------|
| `--embodiment` | Robot: `piper` or `axol` |
| `--episode` | Episode index (default: `0`) |
| `--axis-maps` | Semicolon-separated mappings to compare |
| `--workspace` | `rest` or `front` |

### `bin/process_umi_to_lerobot.sh`

Convert a PICO/UMI LeRobot dataset to embodiment-specific joint angles via IK.

```bash
bash bin/process_umi_to_lerobot.sh \
    --embodiment piper \
    --output-name dexumi-dataset-v2-piper \
    --output-root outputs/datasets/dexumi-dataset-v2-piper
```

| Argument | Description |
|----------|-------------|
| `--repo-id` | Source HuggingFace repo (default: `NONHUMAN-RESEARCH/dexumi-dataset-v2`) |
| `--dataset-root` | Local source dataset path |
| `--embodiment` | Target robot: `piper` or `axol` |
| `--output-name` / `--output-root` | Output dataset name and path |
| `--episodes` | Comma-separated episode indices (default: all) |
| `--push-to-hub` | Upload result to HuggingFace Hub |
