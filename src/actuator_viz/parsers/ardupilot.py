"""
ArduPilot/ArduSub configuration parser.

Parses ArduPilot parameter files and ArduSub motor configurations.

ArduSub uses predefined frame types with motor factor matrices.
This parser supports:
- ArduSub frame type parameters (FRAME_CONFIG)
- Custom motor factor definitions
- Parameter files (.param, .parm)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union
from enum import IntEnum

from .base import ConfigParser
from ..core.models import ActuatorConfig, Actuator, Geometry


class ArduSubFrame(IntEnum):
    """ArduSub predefined frame types."""
    BLUEROV1 = 0
    VECTORED = 1        # BlueROV2 style
    VECTORED_6DOF = 2
    VECTORED_6DOF_90DEG = 3
    SIMPLEROV_3 = 4
    SIMPLEROV_4 = 5
    SIMPLEROV_5 = 6
    CUSTOM = 7


# Predefined ArduSub frame configurations
# Each frame defines motors with [roll, pitch, yaw, throttle, forward, lateral] factors
# These get converted to position/axis format
ARDUSUB_FRAMES = {
    ArduSubFrame.VECTORED: {
        "name": "BlueROV2 (Vectored)",
        "description": "6-thruster vectored frame",
        "motors": [
            # Motor 1: Front Right
            {"position": (0.12, -0.15, 0.0), "axis": (0.707, -0.707, 0.0)},
            # Motor 2: Front Left
            {"position": (0.12, 0.15, 0.0), "axis": (0.707, 0.707, 0.0)},
            # Motor 3: Back Right
            {"position": (-0.12, -0.15, 0.0), "axis": (-0.707, -0.707, 0.0)},
            # Motor 4: Back Left
            {"position": (-0.12, 0.15, 0.0), "axis": (-0.707, 0.707, 0.0)},
            # Motor 5: Vertical Right
            {"position": (0.0, -0.11, 0.0), "axis": (0.0, 0.0, 1.0)},
            # Motor 6: Vertical Left
            {"position": (0.0, 0.11, 0.0), "axis": (0.0, 0.0, 1.0)},
        ],
        "geometry": {"type": "box", "dimensions": (0.45, 0.34, 0.20)},
    },
    ArduSubFrame.VECTORED_6DOF: {
        "name": "Vectored 6DOF",
        "description": "8-thruster full 6DOF frame",
        "motors": [
            # Horizontal thrusters (45 degree)
            {"position": (0.15, -0.15, 0.0), "axis": (0.707, -0.707, 0.0)},
            {"position": (0.15, 0.15, 0.0), "axis": (0.707, 0.707, 0.0)},
            {"position": (-0.15, -0.15, 0.0), "axis": (-0.707, -0.707, 0.0)},
            {"position": (-0.15, 0.15, 0.0), "axis": (-0.707, 0.707, 0.0)},
            # Vertical thrusters
            {"position": (0.10, -0.10, 0.0), "axis": (0.0, 0.0, 1.0)},
            {"position": (0.10, 0.10, 0.0), "axis": (0.0, 0.0, 1.0)},
            {"position": (-0.10, -0.10, 0.0), "axis": (0.0, 0.0, 1.0)},
            {"position": (-0.10, 0.10, 0.0), "axis": (0.0, 0.0, 1.0)},
        ],
        "geometry": {"type": "box", "dimensions": (0.50, 0.40, 0.25)},
    },
    ArduSubFrame.SIMPLEROV_4: {
        "name": "SimpleROV 4",
        "description": "4-thruster simple frame",
        "motors": [
            # 4 vertical thrusters at corners
            {"position": (0.15, -0.15, 0.0), "axis": (0.0, 0.0, 1.0)},
            {"position": (0.15, 0.15, 0.0), "axis": (0.0, 0.0, 1.0)},
            {"position": (-0.15, -0.15, 0.0), "axis": (0.0, 0.0, 1.0)},
            {"position": (-0.15, 0.15, 0.0), "axis": (0.0, 0.0, 1.0)},
        ],
        "geometry": {"type": "box", "dimensions": (0.40, 0.40, 0.15)},
    },
}


class ArduPilotParser(ConfigParser):
    """
    Parser for ArduPilot/ArduSub configuration files.

    Supports:
    - FRAME_CONFIG parameter for predefined frames
    - Custom motor definitions in YAML/param format
    - ArduPilot .param/.parm files
    """

    # Pattern to detect ArduPilot parameters
    PARAM_PATTERN = re.compile(r'^(\w+)\s*[,=]\s*(-?[\d.]+(?:e[+-]?\d+)?)', re.MULTILINE | re.IGNORECASE)

    # Specific patterns
    FRAME_CONFIG_PATTERN = re.compile(r'FRAME_CONFIG\s*[,=]\s*(\d+)', re.IGNORECASE)
    MOT_PATTERN = re.compile(r'MOT_(\d+)_(\w+)\s*[,=]\s*(-?[\d.]+)', re.IGNORECASE)

    @property
    def name(self) -> str:
        return "ArduPilot"

    @property
    def extensions(self) -> list[str]:
        return [".param", ".parm"]

    def can_parse(self, source: Union[str, Path]) -> bool:
        """Check if source is an ArduPilot configuration."""
        content = self._get_content(source)
        if content is None:
            return False

        # Check file extension
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists() and path.suffix.lower() in self.extensions:
                return True

        # Check for ArduPilot-specific parameters
        ardupilot_indicators = [
            'FRAME_CONFIG',
            'MOT_1_',
            'SERVO_FUNCTION',
            'BRD_TYPE',
            'ARMING_CHECK',
        ]

        for indicator in ardupilot_indicators:
            if indicator in content.upper():
                return True

        return False

    def parse(self, source: Union[str, Path]) -> ActuatorConfig:
        """
        Parse ArduPilot configuration to ActuatorConfig.

        Args:
            source: File path or string content

        Returns:
            ActuatorConfig with parsed actuators
        """
        content = self._get_content(source)
        if content is None:
            raise ValueError(f"Could not read source: {source}")

        # Try to detect frame type
        frame_match = self.FRAME_CONFIG_PATTERN.search(content)

        if frame_match:
            frame_type = int(frame_match.group(1))
            return self._parse_predefined_frame(frame_type, source)

        # Try to parse custom motor definitions
        motors = self._parse_motor_params(content)
        if motors:
            return self._build_config_from_motors(motors, source)

        raise ValueError(
            "Could not parse ArduPilot configuration. "
            "No FRAME_CONFIG or motor parameters found."
        )

    def _get_content(self, source: Union[str, Path]) -> str | None:
        """Get string content from file path or string."""
        if isinstance(source, Path):
            if source.exists():
                return source.read_text()
            return None

        if isinstance(source, str):
            path = Path(source)
            if path.exists():
                return path.read_text()
            return source

        return None

    def _parse_predefined_frame(
        self, frame_type: int, source: Union[str, Path]
    ) -> ActuatorConfig:
        """Build config from predefined ArduSub frame type."""
        try:
            frame_enum = ArduSubFrame(frame_type)
        except ValueError:
            raise ValueError(f"Unknown ArduSub frame type: {frame_type}")

        if frame_enum not in ARDUSUB_FRAMES:
            raise ValueError(
                f"Frame type {frame_enum.name} ({frame_type}) not yet supported. "
                f"Supported: {[f.name for f in ARDUSUB_FRAMES.keys()]}"
            )

        frame_def = ARDUSUB_FRAMES[frame_enum]

        actuators = []
        for i, motor in enumerate(frame_def["motors"]):
            actuators.append(Actuator(
                id=i,
                name=f"Motor_{i + 1}",
                position=motor["position"],
                axis=motor["axis"],
                coefficient=1.0,
                moment_ratio=0.0,
            ))

        geometry = None
        if "geometry" in frame_def:
            geo = frame_def["geometry"]
            geometry = Geometry(
                geometry_type=geo["type"],
                dimensions=tuple(geo["dimensions"]),
            )

        name = frame_def["name"]
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                name = f"{frame_def['name']} ({path.stem})"

        return ActuatorConfig(
            name=name,
            actuators=actuators,
            frame="NED",  # ArduPilot uses NED
            units="meters",
            geometry=geometry,
        )

    def _parse_motor_params(self, content: str) -> dict[int, dict]:
        """Parse MOT_*_* parameters from content."""
        motors: dict[int, dict] = {}

        for match in self.MOT_PATTERN.finditer(content):
            motor_num = int(match.group(1))
            param_name = match.group(2).upper()
            value = float(match.group(3))

            if motor_num not in motors:
                motors[motor_num] = {}

            motors[motor_num][param_name] = value

        return motors

    def _build_config_from_motors(
        self, motors: dict[int, dict], source: Union[str, Path]
    ) -> ActuatorConfig:
        """Build ActuatorConfig from parsed motor parameters."""
        actuators = []

        for motor_num, params in sorted(motors.items()):
            # Try to extract position and axis from params
            # ArduPilot motor params vary by vehicle type
            position = (
                params.get('POS_X', params.get('POSX', 0.0)),
                params.get('POS_Y', params.get('POSY', 0.0)),
                params.get('POS_Z', params.get('POSZ', 0.0)),
            )

            axis = (
                params.get('AXIS_X', params.get('DIR_X', 0.0)),
                params.get('AXIS_Y', params.get('DIR_Y', 0.0)),
                params.get('AXIS_Z', params.get('DIR_Z', 1.0)),
            )

            actuators.append(Actuator(
                id=motor_num - 1,  # ArduPilot motors are 1-indexed
                name=f"Motor_{motor_num}",
                position=position,
                axis=axis,
                coefficient=params.get('THRUST_COEF', params.get('CT', 1.0)),
                moment_ratio=params.get('MOMENT_RATIO', params.get('KM', 0.0)),
            ))

        name = "ArduPilot Custom"
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                name = f"ArduPilot: {path.stem}"

        return ActuatorConfig(
            name=name,
            actuators=actuators,
            frame="NED",
            units="meters",
        )


def parse_ardupilot(path: Union[str, Path]) -> ActuatorConfig:
    """
    Parse an ArduPilot parameter file.

    Convenience function for quick parsing.

    Args:
        path: Path to ArduPilot .param file

    Returns:
        ActuatorConfig object
    """
    parser = ArduPilotParser()
    return parser.parse(path)


def get_ardusub_frame(frame_type: int | ArduSubFrame) -> ActuatorConfig:
    """
    Get a predefined ArduSub frame configuration.

    Args:
        frame_type: ArduSubFrame enum or integer

    Returns:
        ActuatorConfig for the frame
    """
    if isinstance(frame_type, int):
        frame_type = ArduSubFrame(frame_type)

    parser = ArduPilotParser()
    # Create a fake param string to trigger predefined frame parsing
    fake_content = f"FRAME_CONFIG,{frame_type.value}"
    return parser.parse(fake_content)


def list_ardusub_frames() -> list[dict]:
    """
    List available predefined ArduSub frames.

    Returns:
        List of dicts with frame info
    """
    frames = []
    for frame_enum, frame_def in ARDUSUB_FRAMES.items():
        frames.append({
            "type": frame_enum.value,
            "name": frame_enum.name,
            "display_name": frame_def["name"],
            "description": frame_def["description"],
            "motor_count": len(frame_def["motors"]),
        })
    return frames
