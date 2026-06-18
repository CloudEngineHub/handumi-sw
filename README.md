# Dexumi

Robotics project built on [LeRobot](https://github.com/huggingface/lerobot) with Feetech motor support, dataset tooling, and XRoboToolkit PC Service bindings.

## Requirements

- Python **>= 3.12**
- [uv](https://docs.astral.sh/uv/) (package and environment manager)
- CMake (required to build `xrobotoolkit_sdk`)

The XRoboToolkit binding is fetched automatically by `bin/install.sh` if it is not already present. The script will:

1. Copy it from `../GR00T-WholeBodyControl/external_dependencies/` when available, or
2. Clone it from [XR-Robotics/XRoboToolkit-PC-Service-Pybind](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind) and build the native SDK for your architecture.

## Installation

All setup is done with **uv**. From the repository root:

```bash
chmod +x bin/install.sh
./bin/install.sh
```

This script is **incremental**: re-running it reuses `.venv`, skips dependencies that are already present, and only installs or updates what changed.

This script will:

1. Ensure `external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/` exists
2. Create a `.venv` with Python >= 3.12 via `uv venv`
3. Install project dependencies via `uv sync`, including:
   - `lerobot[feetech,dataset]==0.5.1` (Feetech motors + dataset extras)
4. Install the local XRoboToolkit binding in editable mode:

```bash
uv pip install --no-build-isolation -e external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/
```

### Manual install

```bash
uv venv --python ">=3.12"
uv sync
uv pip install --no-build-isolation -e external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/
```

Activate the environment:

```bash
source .venv/bin/activate
```

## Dependencies

| Package | Version / extra | Purpose |
|---------|-----------------|---------|
| `lerobot` | `==0.5.1` with `[feetech,dataset]` | Robotics framework, Feetech servo SDK (`feetech-servo-sdk`), dataset tooling |

> **Note:** In `lerobot==0.5.1`, the `feetech` extra installs `feetech-servo-sdk`; `pyserial` and `deepdiff` are already core dependencies. The `dataset` extra is not defined separately in 0.5.1 (dataset-related packages such as `datasets`, `jsonlines`, `av`, and `torchcodec` are included in the base install), but it is listed here for compatibility with newer LeRobot versions.

| `xrobotoolkit_sdk` | local editable | XRoboToolkit PC Service Python bindings |
