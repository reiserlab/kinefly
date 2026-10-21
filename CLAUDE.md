# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Kinefly is a pure Python 3 application that extracts kinematic variables from camera images of tethered winged insects. It tracks head, abdomen, left wing, and right wing movement using computer vision techniques.

## Build & Run

```bash
# Install dependencies
uv sync --extra dev

# Run (headless, with a rig config)
uv run python -m kinefly --rig rigs/example/config.yaml

# Run with ZeroMQ publishing
uv run python -m kinefly --rig rigs/example/config.yaml --zmq tcp://*:5555

# Run with video recording
uv run python -m kinefly --rig rigs/example/config.yaml --record

# Convert a legacy ROS rig to new format
python tools/convert_rig.py launch/thadsrig/ -o rigs/thadsrig/config.yaml

# Diagnostic plots (requires ZeroMQ publisher running)
python tools/plot_tracker.py --address tcp://localhost:5555 --tracker left
python tools/plot_wingdata.py --address tcp://localhost:5555
python tools/plot_tipdata.py --address tcp://localhost:5555 --tracker left

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/ tools/
```

## Data Output

FlyState is emitted via the EventBus after each frame:
- **Callbacks**: register with `EventBus.register(callback)`
- **ZeroMQ**: published as MessagePack-encoded dict at configured address

FlyState fields: `timestamp`, `seq`, `head`, `abdomen`, `left`, `right`, `aux`
Each bodypart: `angles` (rad), `radii` (px), `gradients`, `freq` (Hz), `intensity` [0,1]

## Architecture

### Package structure (`src/kinefly/`)

- **`__main__.py`** — CLI entry point. Wires: config → camera → EventBus → plugins → Fly → loop.
- **`fly.py`** — `Fly` class: instantiates per-bodypart trackers, calls `update()` each frame, emits `FlyState` via EventBus.
- **`core/`** — `types.py` (FlyState, BodyPartState), `events.py` (EventBus + ZeroMQ), `imaging.py` (polar transforms, phase correlation), `setdict.py`
- **`config/`** — `models.py` (dataclasses), `loader.py` (YAML loader with defaults)
- **`camera/`** — `CameraSource` ABC, `OpenCVCamera`, `VideoFileSource`, `HarvesterCamera`
- **`trackers/`** — `base.py`, `area.py`, `edge.py`, `tip.py`, `intensity.py`, `axis.py`, `wingbeat.py`
- **`plugins/`** — `OutputPlugin` base, `PhidgetsPlugin`, `LedPanelsPlugin`
- **`recording/`** — `VideoRecorder` (FFmpeg subprocess)
- **`gui/`** — `ui_colors.py`, `handles.py`, `imagewindow.py`

### Tracker classes

- **`AreaTracker`** — Default for the head. Image registration in polar coordinates.
- **`EdgeTrackerByIntensityProfile`** — Default for wings. Finds radial edges by intensity gradient.
- **`TipTracker`** — Default for the abdomen. Finds the farthest point of a bodypart from its rotation center.
- **`IntensityTracker`** — Monitors pixel intensity; aux region and wingbeat frequency.
- **`AxisTracker`** — Tracks the body axis.

### Configuration

Rig configs are YAML files under `rigs/`. See `rigs/example/config.yaml` for the full schema.

GUI state (hinge positions, checkbox states) auto-saves to `~/kinefly.yaml`.

### Tools (`tools/`)

- **`convert_rig.py`** — Convert legacy ROS `.launch` dirs to new YAML format
- **`plot_tracker.py`** / **`plot_wingdata.py`** / **`plot_tipdata.py`** — Live diagnostic plots via ZeroMQ

## Dependencies

- Python ≥ 3.11
- Core: `numpy`, `opencv-python`, `pyyaml`
- Optional extras (install with `uv sync --extra <name>`):
  - `harvester` — GenICam cameras via `harvesters`
  - `phidgets` — PhidgetsAnalog voltage output via `Phidget22`
  - `ledpanels` — LED panel control via `pyserial`
  - `zmq` — ZeroMQ publishing via `pyzmq` + `msgpack`
  - `plotting` — Diagnostic tools via `matplotlib`
  - `all` — All optional dependencies
  - `dev` — All + `ruff` + `pytest`
