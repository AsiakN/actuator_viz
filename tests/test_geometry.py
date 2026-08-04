"""Tests for core vector/frame geometry utilities."""

from __future__ import annotations

import numpy as np
import pytest

from actuator_viz import cross_product, normalize
from actuator_viz.core.geometry import (
    angle_between_vectors,
    transform_vector,
)
from actuator_viz import CoordinateFrame


def test_cross_product_matches_numpy():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(cross_product(a, b), np.cross(a, b))


def test_cross_product_torque_convention():
    # r = +x, F = +z  =>  r x F = -y  (a pitch-down moment)
    r = np.array([1.0, 0.0, 0.0])
    f = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(cross_product(r, f), [0.0, -1.0, 0.0])


def test_normalize_unit_length():
    v = np.array([0.0, 0.0, 5.0])
    np.testing.assert_allclose(normalize(v), [0.0, 0.0, 1.0])


def test_normalize_zero_vector_is_safe():
    # Must not divide by zero — returns the zero vector.
    np.testing.assert_allclose(normalize(np.zeros(3)), np.zeros(3))


def test_angle_between_orthogonal_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert angle_between_vectors(a, b) == pytest.approx(np.pi / 2)


def test_transform_same_frame_is_identity():
    v = np.array([1.0, 2.0, 3.0])
    out = transform_vector(v, CoordinateFrame.ENU, CoordinateFrame.ENU)
    np.testing.assert_allclose(out, v)


def test_transform_enu_to_ned_roundtrip():
    v = np.array([1.0, 2.0, 3.0])
    ned = transform_vector(v, CoordinateFrame.ENU, CoordinateFrame.NED)
    back = transform_vector(ned, CoordinateFrame.NED, CoordinateFrame.ENU)
    np.testing.assert_allclose(back, v)
    # ENU (E,N,U) -> NED (N,E,D): swap x/y, negate z.
    np.testing.assert_allclose(ned, [2.0, 1.0, -3.0])
