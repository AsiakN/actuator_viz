"""Tests for config parsing and format auto-detection."""

from __future__ import annotations

import pytest

from actuator_viz import CoordinateFrame, parse_config, parse_yaml_string
from actuator_viz.parsers import ArduPilotParser, PX4Parser


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
    # Predefined ArduSub frames warn that their geometry is representative.
    with pytest.warns(UserWarning, match="representative geometry"):
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
    with pytest.raises((FileNotFoundError, ValueError)):
        parse_config(examples_dir / "does_not_exist.yaml")


# --- PX4 parser robustness ---------------------------------------------------

def test_px4_scientific_notation():
    config = PX4Parser().parse("CA_ROTOR0_PX 1.5e-2\nCA_ROTOR0_AZ 1")
    assert config.actuators[0].position[0] == pytest.approx(0.015)


def test_px4_malformed_value_does_not_crash():
    # A bare "." used to crash float(); it must now be skipped, not fatal.
    config = PX4Parser().parse("CA_ROTOR0_PX .\nCA_ROTOR0_PY 0.1\nCA_ROTOR0_AZ 1")
    assert config.actuators[0].position == (0.0, 0.1, 0.0)


def test_px4_plain_set_syntax():
    # `param set` (not set-default) must parse the same way.
    config = PX4Parser().parse("param set CA_ROTOR0_PX 0.2\nparam set CA_ROTOR0_AZ 1")
    assert config.actuators[0].position[0] == pytest.approx(0.2)


def test_px4_stock_airframe_without_geometry_errors():
    # Rotors are referenced (KM) but positions/axes are inherited from defaults,
    # so there's no layout to analyze: expect the geometry-specific error.
    text = (
        "param set-default SYS_AUTOSTART 4001\n"
        "param set-default CA_ROTOR_COUNT 4\n"
        "param set-default CA_ROTOR0_KM 0.05\n"
        "param set-default CA_ROTOR1_KM -0.05\n"
    )
    with pytest.raises(ValueError, match="geometry"):
        PX4Parser().parse(text)


def test_px4_zero_axis_errors():
    text = "CA_ROTOR0_PX 0.1\nCA_ROTOR0_AX 0\nCA_ROTOR0_AY 0\nCA_ROTOR0_AZ 0"
    with pytest.raises(ValueError, match="zero thrust axis"):
        PX4Parser().parse(text)


def test_px4_rotor_count_mismatch_warns():
    text = "CA_ROTOR_COUNT 4\nCA_ROTOR0_PX 0.1\nCA_ROTOR0_AZ 1"
    with pytest.warns(UserWarning, match="CA_ROTOR_COUNT"):
        PX4Parser().parse(text)


# --- ArduPilot parser robustness ---------------------------------------------

def test_ardupilot_does_not_claim_motorless_dump():
    # A plane-style dump (no FRAME_CONFIG, no numbered motor params) must not be
    # falsely claimed just because it has generic ArduPilot markers.
    dump = "ARMING_CHECK,1\nBRD_TYPE,7\nSERVO1_FUNCTION,4\n"
    assert ArduPilotParser().can_parse(dump) is False
    with pytest.raises(ValueError, match="No parser recognized"):
        parse_config(dump)


def test_ardupilot_detected_but_no_geometry_errors(tmp_path):
    # A .param file is claimed by extension; if it has no usable geometry the
    # registry surfaces a clear, actionable error rather than a hard crash.
    p = tmp_path / "plane.param"
    p.write_text("ARMING_CHECK,1\nBRD_TYPE,7\nFENCE_ENABLE,0\n")
    with pytest.raises(ValueError, match="usable ArduPilot geometry|could not build"):
        parse_config(p)


def test_ardupilot_unsupported_frame_errors():
    with pytest.raises(ValueError, match="not yet supported"):
        ArduPilotParser().parse("FRAME_CONFIG,3")


def test_ardupilot_predefined_frame_warns_approximate():
    with pytest.warns(UserWarning, match="representative geometry"):
        config = ArduPilotParser().parse("FRAME_CONFIG,1")
    assert config.n_actuators == 6
