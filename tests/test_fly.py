import copy

import numpy as np
import pytest

from kinefly.core.types import FlyState
from kinefly.fly import Fly


def test_fly_init_default():
    """Fly instantiates with empty params without crashing."""
    fly = Fly(params={})
    assert fly is not None


def test_fly_get_flystate_returns_flystate():
    """get_flystate() returns a FlyState object."""
    fly = Fly(params={})
    state = fly.get_flystate()
    assert isinstance(state, FlyState)
    assert state.seq == 0


def test_fly_get_flystate_increments_seq():
    """Each call to update() increments the sequence counter."""
    fly = Fly(params={})
    img = np.zeros((100, 100), dtype=np.uint8)
    fly.update(img, timestamp=0.1)
    fly.update(img, timestamp=0.2)
    state = fly.get_flystate()
    assert state.seq == 2


def _full_params(width=640, height=480):
    """Complete Fly params: the default rig config and default handle layout."""
    from kinefly.__main__ import _build_fly_params, _default_gui_state
    from kinefly.config.loader import _parse_config

    return _build_fly_params(_parse_config({}), _default_gui_state(width, height), (width, height))


def test_fly_init_sets_body_center():
    """Fly(params) computes the body center from the hinges, as the original did at startup."""
    fly = Fly(params=_full_params())
    np.testing.assert_allclose(fly.ptBodyCenter_i, [300, 200])


def test_fly_auto_invert_color():
    """Auto invert-color runs on the first frame: dark body on a bright background inverts."""
    bright_body = np.zeros((480, 640), dtype=np.uint8)
    bright_body[200:280, 280:360] = 255

    fly = Fly(params=_full_params())
    fly.update(bright_body, timestamp=0.0)
    assert fly.bInvertColorValid
    assert fly.bInvertColor is False

    fly = Fly(params=_full_params())
    fly.update(255 - bright_body, timestamp=0.0)
    assert fly.bInvertColorValid
    assert fly.bInvertColor is True


def _dts(monkeypatch, stamps, clocks):
    """The dt the trackers get for frames with these camera stamps, at these clock times."""
    import kinefly.fly

    clock = iter(clocks)
    monkeypatch.setattr(kinefly.fly.time, "perf_counter", lambda: next(clock))
    fly = Fly(params={})
    dts = []
    fly.head.update = lambda dt, image, bInvertColor: dts.append(dt)
    img = np.zeros((10, 10), dtype=np.uint8)
    for stamp in stamps:
        fly.update(img, timestamp=stamp)
    return dts


def test_dt_from_camera_stamps(monkeypatch):
    assert _dts(monkeypatch, [1.0, 1.01, 1.03], [5.0, 5.5, 6.0]) == pytest.approx(
        [np.inf, 0.01, 0.02]
    )


def test_dt_falls_back_to_clock(monkeypatch):
    """If the camera stamp doesn't advance, dt comes from our own clock, as in the original."""
    assert _dts(monkeypatch, [1.0, 1.0, 0.5], [5.0, 5.25, 5.75]) == pytest.approx(
        [np.inf, 0.25, 0.5]
    )


def test_dt_when_nothing_advances(monkeypatch):
    assert _dts(monkeypatch, [1.0, 1.0], [5.0, 5.0]) == pytest.approx([np.inf, 1.0])


def test_emitted_flystate_is_a_snapshot():
    """A FlyState someone holds on to doesn't change when the next frame is tracked."""
    from kinefly.core.events import EventBus

    params = _full_params()
    for part in ("head", "abdomen", "left", "right", "aux"):
        params["gui"][part]["track"] = True
    states = []
    bus = EventBus()
    bus.register(states.append)
    fly = Fly(params=params, event_bus=bus)

    fly.update(np.full((480, 640), 50, dtype=np.uint8), timestamp=0.0)
    first = copy.deepcopy(states[0])
    fly.update(np.full((480, 640), 200, dtype=np.uint8), timestamp=0.01)

    assert states[1].aux.intensity != first.aux.intensity
    assert states[0] == first
    for part in ("head", "abdomen", "left", "right", "aux"):
        assert getattr(states[0], part) is not getattr(fly, part).state
