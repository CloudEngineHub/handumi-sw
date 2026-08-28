# Installation

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.12.

## System Prerequisites

On Ubuntu/Debian, install these before running `install.sh`. The PICO native
build runs before the virtual environment exists, so its compiler and CMake
must come from the system:

```bash
sudo apt update
sudo apt install -y git curl python3-dev build-essential cmake libportaudio2 adb ffmpeg
```

| Package | Needed for |
| --- | --- |
| `python3-dev` | `Python.h`, required to build `evdev` (pulled in through `lerobot`) |
| `adb` | installing the Quest app and reading its IP; PICO USB diagnostics. `handumi doctor --device pico` fails without it |
| `build-essential`, `cmake` | the PICO `PXREARobotSDK` native build; unused with `--skip-xrt` |
| `libportaudio2` | microphone capture for the default voice control |
| `ffmpeg` | FFmpeg executables and shared libraries loaded by LeRobot/TorchCodec when camera videos are decoded, inspected, or curated |

`ffmpeg` is a system dependency, not a Python package, so `uv sync` cannot
install it. Trajectory replay reads state columns directly and does not decode
camera videos, but recording validation and video inspection still require the
system libraries. A missing or incompatible installation can produce a
`Could not load libtorchcodec` error in those video workflows.

Install `uv` itself if the workstation does not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

## Install HandUMI

```bash
git clone https://github.com/murobotics-ai/handumi-sw.git
cd handumi-sw
bash install.sh              # PICO support included
# bash install.sh --skip-xrt # Meta Quest only
source .venv/bin/activate
```

Check:

```bash
python --version
handumi record --help
```

`install.sh` creates the virtual environment, runs `uv sync`, and builds the
XRoboToolkit SDK needed for PICO. Without `--skip-xrt` it also installs the
XRoboToolkit PC service system package, so it prompts for `sudo` partway
through. Use `--skip-xrt` when the setup only uses Meta Quest, which needs no
PC service and no system compiler. It also creates the ignored machine-local
`configs/rig.yaml` from `configs/rig.example.yaml` without overwriting an
existing rig configuration.
Activating the environment loads command and option completion for Bash, Zsh,
or Fish; for example, `handumi re<Tab>` offers `record` and `replay`.
`hu` is an equivalent short alias for the complete CLI, including help and
completion, so `hu record` and `handumi record` behave identically.

Recording is voice-controlled by default. The speech model is not bundled: the
first `handumi record` downloads it once (~40 MB) to `~/.cache/handumi/vosk/`,
after which recording needs no network. `handumi doctor` reports whether the
microphone and model are ready.

For installations that do not use `install.sh`, enable completion in the
current shell with one of:

```bash
# Bash
eval "$(handumi completion bash)"

# Zsh
eval "$(handumi completion zsh)"

# Fish
handumi completion fish | source
```

## Optional robot and simulation profiles

The base environment does not install manufacturer SDKs. Select only the
profiles needed on the workstation:

```bash
bash install.sh --skip-xrt --sim --robot openarmv1
# Or manage profiles directly after installing system prerequisites:
uv sync --extra sim
uv sync --extra piper
uv sync --extra openarm
uv sync --extra cuda --extra sim
```

Use `bash install.sh --sim` instead of a standalone `uv sync --extra sim` on a
workstation that must retain PICO support. XRoboToolkit is a locally built
package installed after the project sync and is intentionally not part of the
portable lockfile; a later standalone `uv sync` can remove it as an unmanaged
package. Running `install.sh` without `--skip-xrt` performs the sync and then
reinstalls XRoboToolkit. Its removal does not affect dataset replay or
simulation.

`install.sh --robot openarmv1` installs the official Ubuntu system packages
before building the pinned Python binding. The equivalent manual sequence is:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:openarm/main
sudo apt update
sudo apt install -y libopenarm-can-dev openarm-can-utils
uv sync --extra openarm
```

Simulation does not require `piper_sdk` or `openarm_can`.
