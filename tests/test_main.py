"""Tests for the CLI entry point."""

import sys
from pathlib import Path

# Ensure src is on path (uv handles this, but pytest might not)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kinefly.__main__ import _get_version, build_arg_parser


def test_arg_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.rig == Path("rigs/example/config.yaml")
    assert args.headless is False
    assert args.record is False
    assert args.zmq is False
    assert args.verbose is False


def test_arg_parser_rig():
    parser = build_arg_parser()
    args = parser.parse_args(["--rig", "/tmp/my_rig.yaml"])
    assert args.rig == Path("/tmp/my_rig.yaml")


def test_arg_parser_config_alias():
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "/tmp/my_rig.yaml"])
    assert args.rig == Path("/tmp/my_rig.yaml")


def test_arg_parser_zmq():
    parser = build_arg_parser()
    args = parser.parse_args(["--zmq"])
    assert args.zmq is True


def test_arg_parser_record():
    parser = build_arg_parser()
    args = parser.parse_args(["--record"])
    assert args.record is True


def test_arg_parser_verbose():
    parser = build_arg_parser()
    args = parser.parse_args(["-v"])
    assert args.verbose is True


def test_get_version_returns_string():
    v = _get_version()
    assert isinstance(v, str)
    assert len(v) > 0


def test_python_m_kinefly_help(tmp_path):
    """Verify `python -m kinefly --help` exits cleanly."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "kinefly", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0
    assert "kinefly" in result.stdout.lower()


def test_python_m_kinefly_version():
    """Verify `python -m kinefly --version` exits cleanly."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "kinefly", "--version"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    assert result.returncode == 0


def test_load_background(tmp_path):
    """A saved background is loaded if it fits the camera, and ignored if not."""
    import cv2
    import numpy as np

    from kinefly.__main__ import _load_background
    from kinefly.fly import Fly

    path = tmp_path / "kinefly.png"
    img = np.full((48, 64), 77, dtype=np.uint8)
    cv2.imwrite(str(path), img)

    fly = Fly(params={})
    loaded = []
    fly.set_background = loaded.append
    assert _load_background(str(path), fly, (64, 48))
    np.testing.assert_array_equal(loaded[0], img)
    assert not _load_background(str(path), fly, (640, 480))
    assert not _load_background(str(tmp_path / "missing.png"), fly, (64, 48))
    assert len(loaded) == 1
