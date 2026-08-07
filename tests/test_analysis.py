"""Tests for controllability analysis and issue detection."""

from __future__ import annotations

import numpy as np
import pytest

from actuator_viz import (
    Actuator,
    ActuatorConfig,
    analyze,
    analyze_config,
    compute_control_authority,
    compute_effectiveness_matrix,
    detect_issues,
)


def test_identical_actuators_have_rank_one(single_z_thruster):
    # Three copies of the same actuator span a 1-D space.
    config = ActuatorConfig(
        actuators=[Actuator(id=i, position=(0, 0, 0), axis=(0, 0, 1)) for i in range(3)]
    )
    result = analyze(config)
    assert result.rank == 1
    assert not result.controllable


def test_three_axis_forces_rank_three_not_controllable(three_axis_forces):
    result = analyze(three_axis_forces)
    assert result.rank == 3
    assert not result.controllable


def test_rov_is_fully_controllable(rov_config):
    result = analyze(rov_config)
    assert result.rank == 6
    assert result.controllable
    assert np.isfinite(result.condition_number)


def test_condition_number_is_one_for_orthonormal_forces(three_axis_forces):
    # Three orthonormal unit force columns -> all singular values equal -> cond 1.
    result = analyze(three_axis_forces)
    assert result.condition_number == pytest.approx(1.0)


def test_control_authority_sums_absolute_row(three_axis_forces):
    e = compute_effectiveness_matrix(three_axis_forces)
    authority = compute_control_authority(e)
    # Each force axis has exactly one unit contributor.
    assert authority["Fx"] == pytest.approx(1.0)
    assert authority["Fy"] == pytest.approx(1.0)
    assert authority["Fz"] == pytest.approx(1.0)
    assert authority["Roll"] == pytest.approx(0.0)


def test_detect_issues_flags_missing_dof(three_axis_forces):
    e = compute_effectiveness_matrix(three_axis_forces)
    issues = detect_issues(e, three_axis_forces)
    text = " ".join(issues)
    assert "NOT FULLY CONTROLLABLE" in text
    # No actuator contributes to Roll/Pitch/Yaw.
    assert "NO Roll CONTROL" in text


def test_detect_issues_flags_redundant_actuators():
    config = ActuatorConfig(
        actuators=[
            Actuator(id=0, position=(0.0, 0.0, 0.0), axis=(0, 0, 1)),
            Actuator(id=1, position=(0.0, 0.0, 0.0), axis=(0, 0, 1)),
        ]
    )
    e = compute_effectiveness_matrix(config)
    issues = detect_issues(e, config)
    assert any("REDUNDANT" in i for i in issues)


def test_analyze_is_alias_for_analyze_config(three_axis_forces):
    a = analyze(three_axis_forces)
    b = analyze_config(three_axis_forces)
    assert a.rank == b.rank
    assert a.controllable == b.controllable
