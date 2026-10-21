"""Regression test against the oracle: the standalone port of the 2016 Kinefly.

Runs this fork's tracking core offline over every frame of a real recording and
asserts that every value matches the oracle's CSV exactly, at the CSV's precision.
The tracking core is deterministic offline, so any difference is a change in
behaviour.  An intended algorithm change must update the fixtures deliberately;
see data/README.md for where they came from and how to regenerate them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

DATA = Path(__file__).parent / "data"
PARTS = ["head", "abdomen", "left", "right"]

pytestmark = pytest.mark.oracle


def _run_fork(rig_file: Path) -> list[dict[str, str]]:
    """Track every frame of the fixture video; return rows formatted as the oracle's CSV."""
    from kinefly.__main__ import _build_fly_params
    from kinefly.camera.videofile import VideoFileSource
    from kinefly.config.loader import load_rig_config
    from kinefly.core.events import EventBus
    from kinefly.fly import Fly

    config = load_rig_config(rig_file)
    # The oracle saves handles at the top level; this fork expects them under 'gui'.
    with open(DATA / "handles_1440x1080.yaml") as f:
        gui_state = {"gui": yaml.safe_load(f)}

    camera = VideoFileSource(str(DATA / "v001a.mp4"), realtime=False)
    camera.open()
    params = _build_fly_params(config, gui_state, img_size=camera.resolution)
    states = []
    bus = EventBus()
    bus.register(states.append)
    fly = Fly(params=params, event_bus=bus)

    n = max(1, config.tracking.n_edges_max)
    rows = []
    try:
        while True:
            try:
                (frame, stamp) = camera.read()
            except StopIteration:
                break
            fly.update(frame, stamp)
            s = states[-1]  # A live reference: format it before the next update().
            row = {"frame": str(len(rows)), "stamp": f"{stamp:.6f}"}
            for part in PARTS:
                state = getattr(s, part)
                angles = list(state.angles)
                if part in ["left", "right"] and n > 1:
                    for i in range(n):
                        row[f"{part}_angle_{i}"] = f"{angles[i]:.6f}" if i < len(angles) else ""
                else:
                    row[f"{part}_angle"] = f"{angles[0]:.6f}" if angles else ""
                if part in ["head", "abdomen"]:
                    row[f"{part}_radius"] = f"{state.radii[0]:.4f}" if len(state.radii) else ""
            row["aux_intensity"] = f"{s.aux.intensity:.6f}"
            row["aux_freq"] = f"{s.aux.freq:.3f}"
            rows.append(row)
    finally:
        camera.close()
    return rows


@pytest.mark.parametrize("variant", ["example", "walking"])
def test_matches_oracle(variant):
    with open(DATA / f"expected_{variant}.csv", newline="") as f:
        expected = list(csv.DictReader(f))
    actual = _run_fork(DATA / f"rig_{variant}.yaml")

    assert len(actual) == len(expected), f"tracked {len(actual)} frames, oracle has {len(expected)}"
    assert list(actual[0]) == list(expected[0]), "columns differ from the oracle's"

    # Collect every difference, so a failure shows how widespread it is.
    diffs: dict[str, list[tuple[int, str, str]]] = {}
    for i, (a, e) in enumerate(zip(actual, expected)):
        for col in e:
            if a[col] != e[col]:
                diffs.setdefault(col, []).append((i, a[col], e[col]))
    if diffs:
        lines = [f"differs from the oracle ({len(expected)} frames):"]
        for col, d in diffs.items():
            (i, a, e) = d[0]
            lines.append(
                f"  {col}: {len(d)} frames differ; first at frame {i}: got {a}, oracle {e}"
            )
        pytest.fail("\n".join(lines))
