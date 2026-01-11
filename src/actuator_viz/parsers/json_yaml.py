"""
JSON and YAML parser for actuator-viz native configuration format.

Supports the generic actuator configuration schema defined in
schemas/config.schema.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import yaml

from .base import ConfigParser
from ..core.models import ActuatorConfig, Actuator, Geometry, CoordinateFrame, ActuatorType


class JsonYamlParser(ConfigParser):
    """
    Parser for JSON and YAML configuration files.

    Supports the actuator-viz native format:

    ```yaml
    name: "My Robot"
    frame: "ENU"
    units: "meters"

    actuators:
      - id: 0
        name: "Front Left"
        position: [0.1, 0.1, 0]
        axis: [0, 0, 1]
        coefficient: 1.0
        moment_ratio: 0.0

    geometry:
      type: "box"
      dimensions: [1.0, 0.4, 0.3]
    ```
    """

    @property
    def name(self) -> str:
        return "JSON/YAML"

    @property
    def extensions(self) -> list[str]:
        return [".yaml", ".yml", ".json"]

    def can_parse(self, source: Union[str, Path]) -> bool:
        """
        Check if source is a JSON/YAML file or string.

        Args:
            source: File path or string content

        Returns:
            True if this parser can handle the source
        """
        if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
            path = Path(source)
            return path.suffix.lower() in self.extensions

        # Try to detect if it's YAML/JSON content
        if isinstance(source, str):
            source = source.strip()
            # Check for JSON object/array
            if source.startswith("{") or source.startswith("["):
                return True
            # Check for YAML (has colons indicating key-value pairs)
            if ":" in source and ("actuators" in source or "position" in source):
                return True

        return False

    def parse(self, source: Union[str, Path]) -> ActuatorConfig:
        """
        Parse JSON or YAML configuration.

        Args:
            source: File path or string content

        Returns:
            ActuatorConfig object
        """
        data = self._load_data(source)
        return self._parse_data(data)

    def _load_data(self, source: Union[str, Path]) -> dict[str, Any]:
        """Load data from file or string."""
        if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
            path = Path(source)
            content = path.read_text()

            if path.suffix.lower() == ".json":
                return json.loads(content)
            else:
                return yaml.safe_load(content)

        # Assume string content
        source_str = str(source).strip()

        # Try JSON first
        if source_str.startswith("{") or source_str.startswith("["):
            try:
                return json.loads(source_str)
            except json.JSONDecodeError:
                pass

        # Fall back to YAML
        return yaml.safe_load(source_str)

    def _parse_data(self, data: dict[str, Any]) -> ActuatorConfig:
        """Convert parsed data to ActuatorConfig."""
        # Parse actuators
        actuators = []
        for i, act_data in enumerate(data.get("actuators", [])):
            actuator = self._parse_actuator(act_data, i)
            actuators.append(actuator)

        # Parse geometry if present
        geometry = None
        if "geometry" in data:
            geometry = self._parse_geometry(data["geometry"])

        # Parse frame
        frame_str = data.get("frame", "ENU")
        try:
            frame = CoordinateFrame(frame_str)
        except ValueError:
            frame = CoordinateFrame.ENU

        return ActuatorConfig(
            name=data.get("name", "Unnamed Configuration"),
            actuators=actuators,
            frame=frame,
            units=data.get("units", "meters"),
            geometry=geometry,
        )

    def _parse_actuator(self, data: dict[str, Any], index: int) -> Actuator:
        """Parse a single actuator from data."""
        # Handle both new format and legacy format
        position = data.get("position")
        axis = data.get("axis")

        # Legacy format support (px, py, pz, ax, ay, az)
        if position is None:
            position = [
                data.get("px", 0.0),
                data.get("py", 0.0),
                data.get("pz", 0.0),
            ]
        if axis is None:
            axis = [
                data.get("ax", 0.0),
                data.get("ay", 0.0),
                data.get("az", 1.0),
            ]

        # Parse actuator type
        type_str = data.get("type", "thruster")
        try:
            actuator_type = ActuatorType(type_str)
        except ValueError:
            actuator_type = ActuatorType.THRUSTER

        return Actuator(
            id=data.get("id", index),
            name=data.get("name", f"Actuator_{index}"),
            position=tuple(position),
            axis=tuple(axis),
            coefficient=data.get("coefficient", data.get("ct", 1.0)),
            moment_ratio=data.get("moment_ratio", data.get("km", 0.0)),
            actuator_type=actuator_type,
            bidirectional=data.get("bidirectional", True),
        )

    def _parse_geometry(self, data: dict[str, Any]) -> Geometry:
        """Parse geometry data."""
        return Geometry(
            geometry_type=data.get("type", "box"),
            dimensions=tuple(data.get("dimensions", [1.0, 0.4, 0.3])),
            mesh_file=data.get("mesh_file"),
        )


def load_yaml(path: Union[str, Path]) -> ActuatorConfig:
    """
    Load actuator configuration from a YAML file.

    Convenience function for quick loading.

    Args:
        path: Path to YAML file

    Returns:
        ActuatorConfig object
    """
    parser = JsonYamlParser()
    return parser.parse(Path(path))


def load_json(path: Union[str, Path]) -> ActuatorConfig:
    """
    Load actuator configuration from a JSON file.

    Convenience function for quick loading.

    Args:
        path: Path to JSON file

    Returns:
        ActuatorConfig object
    """
    parser = JsonYamlParser()
    return parser.parse(Path(path))


def parse_yaml_string(content: str) -> ActuatorConfig:
    """
    Parse actuator configuration from a YAML string.

    Args:
        content: YAML content

    Returns:
        ActuatorConfig object
    """
    parser = JsonYamlParser()
    return parser.parse(content)
