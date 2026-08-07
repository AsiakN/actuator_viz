"""
Tests for the effectiveness matrix — the physical core of the tool.

Uses hand-computed columns so the tests are independent of the implementation
(they encode the physics, not a re-run of the code under test).

Row order is [Roll, Pitch, Yaw, Fx, Fy, Fz].
"""

from __future__ import annotations

import numpy as np

from actuator_viz import Actuator, ActuatorConfig, compute_effectiveness_matrix


def _matrix(*actuators) -> np.ndarray:
    return compute_effectiveness_matrix(ActuatorConfig(actuators=list(actuators)))


def test_shape_is_six_by_n():
    e = _matrix(
        Actuator(id=0, position=(0, 0, 0), axis=(0, 0, 1)),
        Actuator(id=1, position=(0, 0, 0), axis=(1, 0, 0)),
    )
    assert e.shape == (6, 2)


def test_single_z_thruster_at_origin():
    # Pure +Z force, no torque (zero moment arm).
    e = _matrix(Actuator(id=0, position=(0, 0, 0), axis=(0, 0, 1)))
    np.testing.assert_allclose(e[:, 0], [0, 0, 0, 0, 0, 1], atol=1e-12)


def test_offset_thruster_produces_moment():
    # +Z thrust at r=+x  ->  pitch moment = r x F = -y, plus the Fz force.
    e = _matrix(Actuator(id=0, position=(1, 0, 0), axis=(0, 0, 1)))
    np.testing.assert_allclose(e[:, 0], [0, -1, 0, 0, 0, 1], atol=1e-12)


def test_moment_ratio_creates_yaw_reaction():
    # Spin reaction torque along the thrust axis: yaw = -km, Fz = 1.
    e = _matrix(Actuator(id=0, position=(0, 0, 0), axis=(0, 0, 1), moment_ratio=0.5))
    np.testing.assert_allclose(e[:, 0], [0, 0, -0.5, 0, 0, 1], atol=1e-12)


def test_coefficient_scales_column():
    e1 = _matrix(Actuator(id=0, position=(1, 0, 0), axis=(0, 0, 1), coefficient=1.0))
    e2 = _matrix(Actuator(id=0, position=(1, 0, 0), axis=(0, 0, 1), coefficient=2.0))
    np.testing.assert_allclose(e2[:, 0], 2.0 * e1[:, 0], atol=1e-12)


def test_axis_is_normalized_before_use():
    # A non-unit axis must give the same column as its unit version.
    e_unit = _matrix(Actuator(id=0, position=(0, 0, 0), axis=(0, 0, 1)))
    e_long = _matrix(Actuator(id=0, position=(0, 0, 0), axis=(0, 0, 5)))
    np.testing.assert_allclose(e_long[:, 0], e_unit[:, 0], atol=1e-12)
