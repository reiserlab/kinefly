# Kinefly

Real-time kinematic tracking of tethered winged insects from camera images.

Kinefly extracts body-part motion variables — head, abdomen, left wing, right wing, and an auxiliary region — from a live or recorded camera stream. Several tracking algorithms are available per body part, and all output is accessible via Python callbacks or a ZeroMQ socket for downstream closed-loop experiments.

> **Version 2.x** is a pure Python 3 rewrite. It has no ROS dependency and runs standalone with `uv`. Legacy ROS `.launch` rig configs can be converted with `tools/convert_rig.py`.

---

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- PySide6 (installed automatically with core deps)

## Installation

```bash
git clone https://github.com/elhananby/Kinefly.git
cd Kinefly
uv sync                  # core deps only
uv sync --extra dev      # + ruff, pytest (recommended for development)
uv sync --extra all      # + all optional hardware/plotting deps
```

### Optional extras

| Extra | Installs | When you need it |
|---|---|---|
| `harvester` | `harvesters`, `genicam` | GenICam / GigE cameras |
| `phidgets` | `Phidget22` | PhidgetsAnalog voltage output |
| `ledpanels` | `pyserial` | LED panel serial control |
| `zmq` | `pyzmq`, `msgpack` | ZeroMQ state publishing |
| `plotting` | `matplotlib`, `pyzmq`, `msgpack` | Diagnostic plot tools |
| `all` | everything above | Full rig setup |

---

## Quick start

```bash
# Run with a USB webcam (interactive GUI)
uv run python -m kinefly --rig rigs/example/config.yaml

# Or via the installed script entry point
kinefly --rig rigs/example/config.yaml
```

On first launch (no `~/kinefly.yaml` saved state), default handle positions are placed in the centre of the frame. Drag them onto the fly before enabling tracking — see [Setting up the fly](#setting-up-the-fly) below.

---

## Running

```bash
# GUI mode (default)
uv run python -m kinefly --rig rigs/myrig/config.yaml

# Headless mode — no window, useful on servers or in scripts
uv run python -m kinefly --rig rigs/myrig/config.yaml --headless

# Enable ZeroMQ publishing (overrides rig config zmq block)
uv run python -m kinefly --rig rigs/myrig/config.yaml --zmq

# Publish on a non-default address
uv run python -m kinefly --rig rigs/myrig/config.yaml --zmq tcp://*:5556

# Start recording immediately on launch
uv run python -m kinefly --rig rigs/myrig/config.yaml --record

# Verbose logging
uv run python -m kinefly --rig rigs/myrig/config.yaml -v
```

---

## Setting up the fly

When the GUI opens, an overlay of draggable handles appears on the camera frame.

### Handle layout (wings example)

```
        angle_lo
       /
  hinge ──── angle_hi
       \_____ radius_inner (ring handle)
```

- **hinge** — rotation centre of the body part (drag to the wing base / head pivot)
- **angle_hi / angle_lo** — upper and lower angular limits of the tracking wedge
- **radius_inner** — inner boundary of the wedge (exclude the body)

The **head** and **abdomen** use the same hinge + wedge geometry. The **aux** region uses a centre point + two radius handles (ellipse). The **axis** tracker uses two endpoints.

### Workflow

1. Launch Kinefly. All handles appear at default positions.
2. Drag the **hinge** handle to the correct anatomical pivot.
3. Drag **angle_hi** and **angle_lo** to bracket the region of interest.
4. Drag the **radius_inner** ring to exclude the body.
5. Check **Track H / Track A / Track L / Track R / Track X** in the toolbar to enable tracking for each body part.
6. Handle positions and track states are auto-saved to `~/kinefly.yaml` on every change and on exit.

> **Tip:** Handles are easier to grab than they look — the click target is ~3× larger than the drawn dot.

---

## GUI reference

### Toolbar — Tracking row

| Control | Action |
|---|---|
| **Track H/A/L/R/X** | Enable tracking for head / abdomen / left wing / right wing / aux |
| **Record** | Start/stop MP4 recording to `~/kinefly_recordings/` |
| **Save BG** | Capture the current frame as the background for subtraction |
| **Exit** | Save state and quit |

### Toolbar — Display options row

| Control | Action |
|---|---|
| **SubtBG H/A/L/R/X** | Subtract the saved background from that body part before tracking. Capture a background first with **Save BG**. |
| **InvertColor** | Invert pixel intensities before processing. Use when the fly is lighter than the background. |
| **Windows** | Show per-tracker OpenCV diagnostic windows (useful for debugging tracking quality). |
| **Symmetric** _(disabled)_ | Not yet implemented. Would mirror the left-wing geometry to the right automatically. See tooltip for implementation notes. |

### Keyboard / mouse

| Action | How |
|---|---|
| Drag a handle | Click and hold anywhere within its grab zone, then move |
| Exit | **Exit** button, or close the window |

---

## Configuration

Each experimental rig is described by a single YAML file. See [`rigs/example/config.yaml`](rigs/example/config.yaml) for the full schema.

```yaml
kinefly:
  version: 2

camera:
  source: opencv        # opencv | videofile | harvester
  device_index: 0
  framerate: 60
  scale_image: 1.0      # downsample the frame before processing

tracking:
  n_edges_max: 1        # number of intensity edges to detect per wing
  rc_background: 1000.0 # background subtraction time constant (frames)

  head:
    tracker: area       # area | edge | tip | intensity | axis
    autozero: true      # reset angle origin each frame
    threshold: 0.0
    feathering: 0.0     # Tukey window alpha for polar transform
    saturation_correction: false

  left:
    tracker: edge
    threshold: 0.01

  aux:
    wingbeat_min: 180.0
    wingbeat_max: 220.0

recording:
  output_dir: ~/kinefly_recordings
  encoder: auto         # auto | h264_videotoolbox | h264_nvenc | libx264

gui:
  state_file: ~/kinefly.yaml   # handle positions and checkbox states

# Uncomment to enable ZeroMQ publishing:
# zmq:
#   address: tcp://*:5555
```

A bodypart section only needs the keys that differ from that bodypart's defaults, which are the original Kinefly's: head `area` with feathering 0.0; abdomen `tip`, left and right `edge`, each with feathering 0.25. All default to `autozero: true`, `threshold: 0.0` and `saturation_correction: false`.

### Migrate a legacy ROS rig

```bash
python tools/convert_rig.py launch/thadsrig/ -o rigs/thadsrig/config.yaml
```

---

## Output

`FlyState` is emitted after every frame via the `EventBus`.

### Python callback

```python
from kinefly.core.events import EventBus
from kinefly.core.types import FlyState

bus = EventBus()

def on_state(state: FlyState) -> None:
    print(state.left.angles, state.right.angles)

bus.register(on_state)
```

### ZeroMQ subscriber

```python
import zmq, msgpack

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://localhost:5555")
sub.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    state = msgpack.unpackb(sub.recv(), raw=False)
    print(state["left"]["angles"])
```

### FlyState fields

| Field | Type | Description |
|---|---|---|
| `timestamp` | `float` | Seconds since epoch |
| `seq` | `int` | Frame sequence number |
| `head`, `abdomen`, `left`, `right`, `aux` | `BodyPartState` | Per-bodypart state |

Each `BodyPartState`:

| Field | Unit | Description |
|---|---|---|
| `angles` | radians | Detected edge / rotation angles |
| `radii` | pixels | Radial distance from hinge |
| `gradients` | — | Intensity gradients at detected edges |
| `intensity` | [0, 1] | Mean pixel intensity in ROI |
| `freq` | Hz | Wingbeat frequency (aux only) |

---

## Trackers

| Tracker | Key | Typical use |
|---|---|---|
| `AreaTracker` | `area` | Head (default) — image-registration rotation in polar coordinates |
| `EdgeTrackerByIntensityProfile` | `edge` | Wings (default) — radial intensity gradient edges |
| `TipTracker` | `tip` | Abdomen (default) — farthest point from hinge |
| `IntensityTracker` | `intensity` | Aux region, wingbeat frequency via FFT |
| `AxisTracker` | `axis` | Body axis — two-point line |

---

## Hardware plugins

### PhidgetsAnalog (voltage output)

Add to your rig config:

```yaml
phidgets:
  serial: 0          # 0 = any device
  autorange: false
  channels:
    - enable: true
      coefficients: {l1: 5.0, r1: -5.0}   # L−R on channel 0
    - enable: true
      coefficients: {r1: 5.0}
```

Install: `uv sync --extra phidgets`

### LED panels (serial USB)

```yaml
ledpanels:
  port: /dev/ttyUSB0
  method: usb          # usb | voltage
  mode: velocity       # velocity | position
  pattern_id: 1
  coeff_usb:
    xl1: 1.0           # x velocity = left wing angle
    xr1: -1.0          #            − right wing angle
```

Install: `uv sync --extra ledpanels`

---

## Diagnostic tools

These tools require a running Kinefly instance with ZeroMQ enabled (`--zmq`).

```bash
# Single body-part: intensity + angle time-series
python tools/plot_tracker.py --address tcp://localhost:5555 --tracker left

# Both wings: angle + intensity
python tools/plot_wingdata.py --address tcp://localhost:5555

# Tip tracker: angle + radius
python tools/plot_tipdata.py --address tcp://localhost:5555 --tracker left
```

Install: `uv sync --extra plotting`

---

## Development

```bash
uv sync --extra dev

# Tests
uv run pytest

# Lint
uv run ruff check src/ tests/ tools/

# Format
uv run ruff format src/ tests/ tools/
```

---

## Project structure

```
src/kinefly/
├── __main__.py         # CLI entry point; wires config → camera → fly → loop
├── app.py              # KineflyApp — PySide6 QTimer-driven frame loop
├── fly.py              # Fly — orchestrates all trackers, emits FlyState
├── core/
│   ├── types.py        # FlyState, BodyPartState dataclasses
│   ├── events.py       # EventBus (callbacks + optional ZeroMQ publisher)
│   ├── imaging.py      # Polar transforms, phase correlation, Tukey window
│   └── setdict.py      # SetDict utility (recursive dict merge)
├── config/
│   ├── models.py       # RigConfig dataclasses
│   └── loader.py       # YAML loader with defaults
├── camera/
│   ├── base.py         # CameraSource ABC
│   ├── opencv.py       # OpenCVCamera (webcam / USB)
│   ├── videofile.py    # VideoFileSource (pre-recorded video)
│   └── harvester.py    # HarvesterCamera (GenICam / GigE)
├── trackers/
│   ├── base.py         # MotionTrackedBodypart, IntensityTrackedBodypart, AxisTracker
│   ├── area.py         # AreaTracker — phase-correlation rotation
│   ├── edge.py         # EdgeTrackerByIntensityProfile — radial gradient edges
│   ├── tip.py          # TipTracker — farthest point from hinge
│   ├── intensity.py    # IntensityTracker — mean intensity + wingbeat FFT
│   ├── axis.py         # AxisTracker — two-point body axis
│   └── wingbeat.py     # Wingbeat frequency estimation
├── plugins/
│   ├── base.py         # OutputPlugin ABC
│   ├── phidgets.py     # PhidgetsPlugin — voltage output
│   └── ledpanels.py    # LedPanelsPlugin — serial LED panel control
├── recording/
│   └── recorder.py     # VideoRecorder — FFmpeg subprocess (h264, auto encoder)
└── gui/
    ├── ui_colors.py    # BGRA color constants, draw_scale() helper
    ├── handles.py      # Handle — draggable overlay point with scaled hit zone
    ├── imagewindow.py  # OpenCV imshow diagnostic windows
    ├── widgets.py      # ImageLabel — QLabel with mouse signals + coord mapping
    └── window.py       # MainWindow — toolbar, status bar, image display

tools/
├── convert_rig.py      # Migrate legacy ROS .launch dirs to YAML
├── plot_tracker.py     # Live diagnostic: single body part
├── plot_wingdata.py    # Live diagnostic: both wings
└── plot_tipdata.py     # Live diagnostic: tip tracker

rigs/
└── example/
    └── config.yaml     # Annotated example rig config
```

---

## GUI state file

Handle positions and checkbox states are persisted to `~/kinefly.yaml` (configurable via `gui.state_file` in the rig config). The file is written automatically whenever a handle is moved or a checkbox is toggled.

```yaml
gui:
  head:
    hinge: {x: 320, y: 240}
    angle_hi: 1.05
    angle_lo: -1.05
    radius_inner: 60.0
    radius_outer: 120.0
    track: true
    subtract_bg: false
    stabilize: false
  left:
    # ... same structure
  aux:
    center: {x: 320, y: 240}
    radius1: 40.0
    radius2: 60.0
    angle: 0.0
    track: false
    subtract_bg: false
  axis:
    pt1: {x: 320, y: 120}
    pt2: {x: 320, y: 360}
    track: false
  windows: false
  invert_color: false
```

---

## License

MIT
