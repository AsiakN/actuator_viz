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
import warnings
from enum import IntEnum
from pathlib import Path

from ..core.models import Actuator, ActuatorConfig, Geometry
from .base import NUMBER_RE, ConfigParser


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

    # Specific patterns
    FRAME_CONFIG_PATTERN = re.compile(r'FRAME_CONFIG\s*[,=]\s*(\d+)', re.IGNORECASE)
    MOT_PATTERN = re.compile(rf'MOT_(\d+)_(\w+)\s*[,=]\s*({NUMBER_RE})', re.IGNORECASE)

    # Per-motor keys that actually describe a layout (position / axis).
    _GEOMETRY_KEYS = frozenset({
        'POS_X', 'POSX', 'POS_Y', 'POSY', 'POS_Z', 'POSZ',
        'AXIS_X', 'DIR_X', 'AXIS_Y', 'DIR_Y', 'AXIS_Z', 'DIR_Z',
    })

    @property
    def name(self) -> str:
        return "ArduPilot"

    @property
    def extensions(self) -> list[str]:
        return [".param", ".parm"]

    def can_parse(self, source: str | Path) -> bool:
        """Check if source is an ArduPilot configuration."""
        content = self._get_content(source)
        if content is None:
            return False

        # A .param/.parm file is ours to attempt (parse() reports clearly if it
        # turns out to carry no usable geometry).
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists() and path.suffix.lower() in self.extensions:
                return True

        # For raw content, only claim it on signals we can actually act on: an
        # ArduSub frame selector or numbered motor params. Generic markers like
        # ARMING_CHECK / BRD_TYPE appear in every ArduPilot dump (including
        # motorless planes) and would falsely claim files we can't parse.
        return bool(
            self.FRAME_CONFIG_PATTERN.search(content)
            or self.MOT_PATTERN.search(content)
        )

    def parse(self, source: str | Path) -> ActuatorConfig:
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

        # Try to parse custom motor definitions — but only if they actually
        # carry position/axis data. Numbered motor params without geometry
        # (e.g. MOT_1_DIRECTION) would otherwise produce motors stacked at the
        # origin, which is worse than a clear error.
        motors = self._parse_motor_params(content)
        if motors and any(self._GEOMETRY_KEYS & p.keys() for p in motors.values()):
            return self._build_config_from_motors(motors, source)

        raise ValueError(
            "No usable ArduPilot geometry found. Supported: an ArduSub "
            f"FRAME_CONFIG frame ({[f.name for f in ARDUSUB_FRAMES]}), or "
            "numbered motor params with positions/axes (MOT_n_POS_X, "
            "MOT_n_AXIS_X, ...). A standard vehicle parameter dump doesn't "
            "encode motor positions — describe the layout in a YAML config."
        )

    def _get_content(self, source: str | Path) -> str | None:
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
        self, frame_type: int, source: str | Path
    ) -> ActuatorConfig:
        """Build config from predefined ArduSub frame type."""
        try:
            frame_enum = ArduSubFrame(frame_type)
        except ValueError as err:
            raise ValueError(f"Unknown ArduSub frame type: {frame_type}") from err

        if frame_enum not in ARDUSUB_FRAMES:
            raise ValueError(
                f"Frame type {frame_enum.name} ({frame_type}) not yet supported. "
                f"Supported: {[f.name for f in ARDUSUB_FRAMES.keys()]}"
            )

        frame_def = ARDUSUB_FRAMES[frame_enum]

        # ArduSub defines motors by control-mix factors, not Cartesian mounts.
        # The positions/axes here are a representative layout for that frame
        # type, not this vehicle's measured geometry — say so.
        warnings.warn(
            f"Using a representative geometry for the '{frame_def['name']}' "
            "ArduSub frame; verify positions/axes against your actual vehicle.",
            stacklevel=2,
        )

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
        self, motors: dict[int, dict], source: str | Path
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


def parse_ardupilot(path: str | Path) -> ActuatorConfig:
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
