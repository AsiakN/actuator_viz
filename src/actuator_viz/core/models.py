"""
Data models for actuator configurations.

Defines the core dataclasses used throughout actuator-viz:
- Actuator: Single actuator (thruster, motor, servo)
- ActuatorConfig: Complete system configuration
- AnalysisResult: Output from controllability analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class CoordinateFrame(str, Enum):
    """Coordinate frame convention."""
    ENU = "ENU"  # East-North-Up (ROS standard)
    NED = "NED"  # North-East-Down (aviation/PX4 standard)
    BODY = "body"  # Body-fixed frame


class ActuatorType(str, Enum):
    """Type of actuator."""
    THRUSTER = "thruster"
    PROPELLER = "propeller"
    SERVO = "servo"
    MOTOR = "motor"


@dataclass
class Actuator:
    """
    Single actuator configuration.

    Attributes:
        id: Unique identifier (usually index)
        name: Human-readable name
        position: [x, y, z] position relative to CoG in meters
        axis: [x, y, z] thrust/force direction (will be normalized)
        coefficient: Thrust/force coefficient (default 1.0)
        moment_ratio: Yaw moment from spin (km), 0 for most thrusters
        actuator_type: Type of actuator
        bidirectional: Whether actuator can produce force in both directions
    """
    id: int
    position: tuple[float, float, float]
    axis: tuple[float, float, float]
    name: str = ""
    coefficient: float = 1.0
    moment_ratio: float = 0.0
    actuator_type: ActuatorType = ActuatorType.THRUSTER
    bidirectional: bool = True

    def __post_init__(self):
        # Convert to tuples if lists were passed
        if isinstance(self.position, list):
            self.position = tuple(self.position)
        if isinstance(self.axis, list):
            self.axis = tuple(self.axis)

        # Validate
        if len(self.position) != 3:
            raise ValueError(f"Position must have 3 components, got {len(self.position)}")
        if len(self.axis) != 3:
            raise ValueError(f"Axis must have 3 components, got {len(self.axis)}")

        # Check axis is not zero vector
        axis_norm = np.linalg.norm(self.axis)
        if axis_norm < 1e-10:
            raise ValueError(f"Axis cannot be zero vector for actuator {self.id}")

        # Auto-generate name if not provided
        if not self.name:
            self.name = f"Actuator_{self.id}"

    @property
    def position_array(self) -> np.ndarray:
        """Position as numpy array."""
        return np.array(self.position)

    @property
    def axis_array(self) -> np.ndarray:
        """Axis as numpy array."""
        return np.array(self.axis)

    @property
    def axis_normalized(self) -> np.ndarray:
        """Normalized axis vector."""
        axis = self.axis_array
        return axis / np.linalg.norm(axis)

    def to_dict(self) -> dict:
        """Convert to dictionary (for serialization)."""
        return {
            "id": self.id,
            "name": self.name,
            "position": list(self.position),
            "axis": list(self.axis),
            "coefficient": self.coefficient,
            "moment_ratio": self.moment_ratio,
            "type": self.actuator_type.value,
            "bidirectional": self.bidirectional,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Actuator:
        """Create from dictionary."""
        actuator_type = data.get("type", "thruster")
        if isinstance(actuator_type, str):
            actuator_type = ActuatorType(actuator_type)

        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            position=tuple(data.get("position", [0, 0, 0])),
            axis=tuple(data.get("axis", [0, 0, 1])),
            coefficient=data.get("coefficient", 1.0),
            moment_ratio=data.get("moment_ratio", 0.0),
            actuator_type=actuator_type,
            bidirectional=data.get("bidirectional", True),
        )

    @classmethod
    def from_rotor_dict(cls, rotor: dict, index: int = 0) -> Actuator:
        """
        Create from legacy rotor dict format (px, py, pz, ax, ay, az).

        This supports the format used by PX4 and the original effectiveness_calculator.
        """
        return cls(
            id=rotor.get("id", index),
            name=rotor.get("name", f"Rotor_{index}"),
            position=(
                rotor.get("px", 0.0),
                rotor.get("py", 0.0),
                rotor.get("pz", 0.0),
            ),
            axis=(
                rotor.get("ax", 0.0),
                rotor.get("ay", 0.0),
                rotor.get("az", 1.0),
            ),
            coefficient=rotor.get("ct", 1.0),
            moment_ratio=rotor.get("km", 0.0),
        )


@dataclass
class Geometry:
    """
    Vehicle geometry for visualization.

    Attributes:
        geometry_type: Type of geometry (box, cylinder, mesh)
        dimensions: Dimensions based on type
            - box: [length, width, height]
            - cylinder: [radius, height]
            - mesh: not used
        mesh_file: Path to STL/OBJ file (for mesh type)
    """
    geometry_type: str = "box"
    dimensions: tuple[float, ...] = (1.0, 0.4, 0.3)
    mesh_file: Optional[Path] = None

    def __post_init__(self):
        if isinstance(self.dimensions, list):
            self.dimensions = tuple(self.dimensions)
        if isinstance(self.mesh_file, str):
            self.mesh_file = Path(self.mesh_file)


@dataclass
class ActuatorConfig:
    """
    Complete actuator system configuration.

    Attributes:
        name: Configuration name
        actuators: List of actuator definitions
        frame: Coordinate frame convention
        units: Position units (meters, millimeters, etc.)
        geometry: Optional vehicle geometry for visualization
    """
    actuators: list[Actuator]
    name: str = "Unnamed Configuration"
    frame: CoordinateFrame = CoordinateFrame.ENU
    units: str = "meters"
    geometry: Optional[Geometry] = None

    def __post_init__(self):
        if isinstance(self.frame, str):
            self.frame = CoordinateFrame(self.frame)

        # Validate actuator IDs are unique
        ids = [a.id for a in self.actuators]
        if len(ids) != len(set(ids)):
            raise ValueError("Actuator IDs must be unique")

    @property
    def n_actuators(self) -> int:
        """Number of actuators."""
        return len(self.actuators)

    def get_actuator(self, id_or_name: int | str) -> Optional[Actuator]:
        """Get actuator by ID or name."""
        for actuator in self.actuators:
            if actuator.id == id_or_name or actuator.name == id_or_name:
                return actuator
        return None

    def to_dict(self) -> dict:
        """Convert to dictionary (for serialization)."""
        result = {
            "name": self.name,
            "frame": self.frame.value,
            "units": self.units,
            "actuators": [a.to_dict() for a in self.actuators],
        }
        if self.geometry:
            result["geometry"] = {
                "type": self.geometry.geometry_type,
                "dimensions": list(self.geometry.dimensions),
            }
            if self.geometry.mesh_file:
                result["geometry"]["mesh_file"] = str(self.geometry.mesh_file)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ActuatorConfig:
        """Create from dictionary."""
        actuators = [
            Actuator.from_dict(a) for a in data.get("actuators", [])
        ]

        geometry = None
        if "geometry" in data:
            geo_data = data["geometry"]
            geometry = Geometry(
                geometry_type=geo_data.get("type", "box"),
                dimensions=tuple(geo_data.get("dimensions", [1.0, 0.4, 0.3])),
                mesh_file=geo_data.get("mesh_file"),
            )

        return cls(
            name=data.get("name", "Unnamed Configuration"),
            actuators=actuators,
            frame=data.get("frame", "ENU"),
            units=data.get("units", "meters"),
            geometry=geometry,
        )

    @classmethod
    def from_rotor_list(cls, rotors: list[dict], name: str = "Imported Config") -> ActuatorConfig:
        """
        Create from legacy rotor list format.

        This supports the format used by PX4 and the original effectiveness_calculator.
        """
        actuators = [
            Actuator.from_rotor_dict(r, i) for i, r in enumerate(rotors)
        ]
        return cls(name=name, actuators=actuators)

    def to_rotor_list(self) -> list[dict]:
        """
        Convert to legacy rotor list format.

        Returns list of dicts with px, py, pz, ax, ay, az, ct, km keys.
        """
        return [
            {
                "px": a.position[0],
                "py": a.position[1],
                "pz": a.position[2],
                "ax": a.axis[0],
                "ay": a.axis[1],
                "az": a.axis[2],
                "ct": a.coefficient,
                "km": a.moment_ratio,
            }
            for a in self.actuators
        ]


@dataclass
class AnalysisResult:
    """
    Result from controllability analysis.

    Attributes:
        rank: Matrix rank (6 = full controllability)
        controllable: True if rank == 6
        singular_values: SVD singular values (control authority per axis)
        condition_number: Ratio of max/min singular values
        weak_axes: List of axes with low control authority
        effectiveness_matrix: The 6xN effectiveness matrix
        U: Left singular vectors from SVD
        issues: List of detected configuration issues
    """
    rank: int
    controllable: bool
    singular_values: np.ndarray
    condition_number: float
    weak_axes: list[str] = field(default_factory=list)
    effectiveness_matrix: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary (for serialization)."""
        return {
            "rank": self.rank,
            "controllable": self.controllable,
            "singular_values": self.singular_values.tolist(),
            "condition_number": self.condition_number,
            "weak_axes": self.weak_axes,
            "issues": self.issues,
        }
