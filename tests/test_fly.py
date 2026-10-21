import numpy as np

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
    np.testing.assert_allclose(fly.ptBodyCenter_i, [320, 240])


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
