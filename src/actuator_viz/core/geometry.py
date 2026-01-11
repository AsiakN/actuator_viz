"""
Geometry utilities for actuator calculations.

Provides vector math and coordinate frame transformations.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from .models import CoordinateFrame


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cross product a × b.

    Cross product formula:
        (a × b)_x = a_y * b_z - a_z * b_y
        (a × b)_y = a_z * b_x - a_x * b_z
        (a × b)_z = a_x * b_y - a_y * b_x

    For torque calculation: τ = r × F
    where r is position vector and F is force vector.

    Args:
        a: 3D vector [x, y, z]
        b: 3D vector [x, y, z]

    Returns:
        3D vector perpendicular to both a and b
    """
    return np.array([
        a[1] * b[2] - a[2] * b[1],  # x component
        a[2] * b[0] - a[0] * b[2],  # y component
        a[0] * b[1] - a[1] * b[0],  # z component
    ])


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Args:
        v: Input vector

    Returns:
        Unit vector in same direction, or zero vector if input is zero
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm


def rotation_matrix_enu_to_ned() -> np.ndarray:
    """
    Get rotation matrix to convert from ENU to NED frame.

    ENU: East-North-Up (x=East, y=North, z=Up)
    NED: North-East-Down (x=North, y=East, z=Down)

    Transformation:
        x_ned = y_enu  (North)
        y_ned = x_enu  (East)
        z_ned = -z_enu (Down)

    Returns:
        3x3 rotation matrix
    """
    return np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ], dtype=float)


def rotation_matrix_ned_to_enu() -> np.ndarray:
    """
    Get rotation matrix to convert from NED to ENU frame.

    Returns:
        3x3 rotation matrix (transpose of ENU->NED)
    """
    return rotation_matrix_enu_to_ned().T


def transform_vector(
    v: np.ndarray,
    from_frame: CoordinateFrame,
    to_frame: CoordinateFrame
) -> np.ndarray:
    """
    Transform a vector between coordinate frames.

    Args:
        v: 3D vector to transform
        from_frame: Source coordinate frame
        to_frame: Target coordinate frame

    Returns:
        Transformed vector
    """
    if from_frame == to_frame:
        return v.copy()

    if from_frame == CoordinateFrame.ENU and to_frame == CoordinateFrame.NED:
        return rotation_matrix_enu_to_ned() @ v
    elif from_frame == CoordinateFrame.NED and to_frame == CoordinateFrame.ENU:
        return rotation_matrix_ned_to_enu() @ v
    else:
        # BODY frame requires vehicle orientation, just return as-is for now
        # get orientation based on position of 3D visualization
        return v.copy()


def rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create rotation matrix from axis-angle representation.

    Uses Rodrigues' rotation formula.

    Args:
        axis: Rotation axis (will be normalized)
        angle: Rotation angle in radians

    Returns:
        3x3 rotation matrix
    """
    axis = normalize(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    x, y, z = axis

    return np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])


def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate angle between two vectors in radians.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Angle in radians [0, π]
    """
    a_norm = normalize(a)
    b_norm = normalize(b)
    dot = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    return np.arccos(dot)


def project_onto_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Project a vector onto a plane defined by its normal.

    Args:
        v: Vector to project
        normal: Plane normal (will be normalized)

    Returns:
        Projected vector lying in the plane
    """
    n = normalize(normal)
    return v - np.dot(v, n) * n
