"""Tests for config parsing and format auto-detection."""

from __future__ import annotations

from actuator_viz import CoordinateFrame, parse_config, parse_yaml_string


def test_parse_yaml_example(examples_dir):
    config = parse_config(examples_dir / "rov_8_thruster.yaml")
    assert config.name == "UUV Reconbot (Fixed)"
    assert config.n_actuators == 8
    assert config.frame is CoordinateFrame.ENU


def test_autodetect_px4_airframe(examples_dir):
    config = parse_config(examples_dir / "px4_quadcopter")
    assert config.n_actuators == 4
    assert config.frame is CoordinateFrame.NED


def test_autodetect_ardupilot_params(examples_dir):
    config = parse_config(examples_dir / "ardusub_bluerov2.param")
    assert config.n_actuators == 6
    assert config.frame is CoordinateFrame.NED


def test_parse_yaml_string_roundtrip():
    from textwrap import dedent

    text = dedent(
        """
        name: "Inline Test"
        frame: "ENU"
        actuators:
          - id: 0
            position: [0.0, 0.0, 0.0]
            axis: [0, 0, 1]
          - id: 1
            position: [1.0, 0.0, 0.0]
            axis: [0, 0, 1]
        """
    )
    config = parse_yaml_string(text)
    assert config.name == "Inline Test"
    assert config.n_actuators == 2


def test_missing_file_raises(examples_dir):
    import pytest

    with pytest.raises((FileNotFoundError, ValueError)):
        parse_config(examples_dir / "does_not_exist.yaml")
