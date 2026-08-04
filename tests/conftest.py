"""Shared fixtures for the actuator-viz test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from actuator_viz import Actuator, ActuatorConfig

# Directory holding the bundled example configs.
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def examples_dir() -> Path:
    """Path to the bundled example configurations."""
    return EXAMPLES


@pytest.fixture
def single_z_thruster() -> ActuatorConfig:
    """One thruster at the origin pointing +Z. Controls Fz only."""
    return ActuatorConfig(
        name="single-z",
        actuators=[Actuator(id=0, position=(0, 0, 0), axis=(0, 0, 1))],
    )


@pytest.fixture
def three_axis_forces() -> ActuatorConfig:
    """
    Three thrusters at the origin along X, Y, Z.

    Produces pure forces on each axis and no torque, so the achievable
    DOFs are exactly {Fx, Fy, Fz} and the rank is 3 (not controllable).
    """
    return ActuatorConfig(
        name="three-axis",
        actuators=[
            Actuator(id=0, position=(0, 0, 0), axis=(1, 0, 0)),
            Actuator(id=1, position=(0, 0, 0), axis=(0, 1, 0)),
            Actuator(id=2, position=(0, 0, 0), axis=(0, 0, 1)),
        ],
    )


@pytest.fixture
def forces_with_redundant_z() -> ActuatorConfig:
    """
    Fx, Fy, and two redundant Fz thrusters (all at the origin).

    Baseline achievable DOFs are {Fx, Fy, Fz}. Actuator 0 (Fx) is the sole
    source of Fx, so its loss is critical. The two Fz thrusters (ids 2 and 3)
    back each other up, so either one failing is non-critical.
    """
    return ActuatorConfig(
        name="redundant-z",
        actuators=[
            Actuator(id=0, position=(0, 0, 0), axis=(1, 0, 0)),  # Fx (unique)
            Actuator(id=1, position=(0, 0, 0), axis=(0, 1, 0)),  # Fy (unique)
            Actuator(id=2, position=(0, 0, 0), axis=(0, 0, 1)),  # Fz
            Actuator(id=3, position=(0, 0, 0), axis=(0, 0, 1)),  # Fz (redundant)
        ],
    )


@pytest.fixture
def rov_config(examples_dir):
    """The bundled 8-thruster ROV — fully controllable and fault tolerant."""
    from actuator_viz import parse_config

    return parse_config(examples_dir / "rov_8_thruster.yaml")
