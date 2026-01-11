"""
Effectiveness matrix computation.

Computes the 6×N effectiveness matrix that maps actuator commands
to forces and torques on the vehicle.
"""

from __future__ import annotations

import numpy as np

from .models import Actuator, ActuatorConfig
from .geometry import cross_product


def compute_effectiveness_matrix(config: ActuatorConfig) -> np.ndarray:
    """
    Compute 6×N effectiveness matrix from actuator configuration.

    The effectiveness matrix maps actuator commands to forces and torques:
        [roll, pitch, yaw, Fx, Fy, Fz]^T = E × [u0, u1, ..., uN]^T

    For each actuator:
        thrust = coefficient × axis
        torque = coefficient × (position × axis) - coefficient × moment_ratio × axis

    The torque has two components:
    1. Moment from force at a distance: τ = r × F
    2. Reaction torque from spinning (propellers): τ = -km × F

    Args:
        config: ActuatorConfig with list of actuators

    Returns:
        6×N numpy array where:
            Rows 0-2: Roll, Pitch, Yaw torque contributions
            Rows 3-5: Fx, Fy, Fz thrust contributions
    """
    n_actuators = config.n_actuators
    effectiveness = np.zeros((6, n_actuators))

    for i, actuator in enumerate(config.actuators):
        position = actuator.position_array
        axis = actuator.axis_normalized
        ct = actuator.coefficient
        km = actuator.moment_ratio

        # Compute thrust contribution (force in axis direction)
        thrust = ct * axis

        # Compute torque contribution
        # 1. Moment from force at position: τ = r × F = position × (ct × axis)
        moment_from_position = ct * cross_product(position, axis)

        # 2. Reaction torque from spinning (e.g., propeller)
        # This creates a torque along the thrust axis
        moment_from_spin = ct * km * axis

        # Total torque (subtract spin moment as it opposes rotation)
        torque = moment_from_position - moment_from_spin

        # Fill effectiveness matrix
        # Torques (rows 0-2)
        effectiveness[0, i] = torque[0]  # Roll
        effectiveness[1, i] = torque[1]  # Pitch
        effectiveness[2, i] = torque[2]  # Yaw

        # Forces (rows 3-5)
        effectiveness[3, i] = thrust[0]  # Fx
        effectiveness[4, i] = thrust[1]  # Fy
        effectiveness[5, i] = thrust[2]  # Fz

    return effectiveness


def compute_effectiveness_from_rotors(rotors: list[dict]) -> np.ndarray:
    """
    Compute effectiveness matrix from legacy rotor dict format.

    This is a convenience function for compatibility with existing code
    that uses the {px, py, pz, ax, ay, az, ct, km} format.

    Args:
        rotors: List of rotor dicts with position and axis keys

    Returns:
        6×N effectiveness matrix
    """
    config = ActuatorConfig.from_rotor_list(rotors)
    return compute_effectiveness_matrix(config)


def get_dof_names() -> list[str]:
    """Get names of the 6 degrees of freedom."""
    return ["Roll", "Pitch", "Yaw", "Fx", "Fy", "Fz"]


def get_torque_dof_names() -> list[str]:
    """Get names of torque DOFs (first 3 rows)."""
    return ["Roll", "Pitch", "Yaw"]


def get_force_dof_names() -> list[str]:
    """Get names of force DOFs (last 3 rows)."""
    return ["Fx", "Fy", "Fz"]


def print_effectiveness_matrix(
    effectiveness: np.ndarray,
    actuator_names: list[str] | None = None
) -> None:
    """
    Pretty-print the effectiveness matrix to console.

    Args:
        effectiveness: 6×N effectiveness matrix
        actuator_names: Optional list of actuator names
    """
    n_actuators = effectiveness.shape[1]
    dof_names = get_dof_names()

    if actuator_names is None:
        actuator_names = [f"A{i}" for i in range(n_actuators)]

    # Header
    print("\n" + "=" * 80)
    print("EFFECTIVENESS MATRIX (6 × N)")
    print("=" * 80)

    # Column headers
    header = f"{'DOF':<12}"
    for name in actuator_names:
        header += f"{name:>10}"
    print(header)
    print("-" * 80)

    # Matrix rows
    for i, dof in enumerate(dof_names):
        row = f"{dof:<12}"
        for j in range(n_actuators):
            val = effectiveness[i, j]
            if abs(val) < 1e-6:
                row += f"{'0':>10}"
            else:
                row += f"{val:>10.4f}"
        print(row)

    print("=" * 80)
