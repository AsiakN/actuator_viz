"""
PX4 airframe file parser.

Parses PX4 airframe configuration files that contain CA_ROTOR parameters.

Example PX4 airframe format:
    param set-default CA_ROTOR_COUNT 8
    param set-default CA_ROTOR0_PX -0.42448
    param set-default CA_ROTOR0_PY 0.1339
    param set-default CA_ROTOR0_PZ -0.1167
    param set-default CA_ROTOR0_AX 0
    param set-default CA_ROTOR0_AY -0.70710678
    param set-default CA_ROTOR0_AZ 0.70710678
    param set-default CA_ROTOR0_KM 0
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

from .base import ConfigParser
from ..core.models import ActuatorConfig, Actuator


class PX4Parser(ConfigParser):
    """
    Parser for PX4 airframe configuration files.

    Extracts CA_ROTOR parameters to build actuator configurations.
    Supports both `param set-default` and `param set` syntax.
    """

    # Regex patterns for CA_ROTOR parameters
    # Matches: CA_ROTOR0_PX, CA_ROTOR10_AZ, etc.
    PARAM_PATTERNS = {
        'px': re.compile(r'CA_ROTOR(\d+)_PX\s+(-?[\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE),
        'py': re.compile(r'CA_ROTOR(\d+)_PY\s+(-?[\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE),
        'pz': re.compile(r'CA_ROTOR(\d+)_PZ\s+(-?[\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE),
        'ax': re.compile(r'CA_ROTOR(\d+)_AX\s+(-?[\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE),
        'ay': re.compile(r'CA_ROTOR(\d+)_AY\s+(-?[\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE),
        'az': re.compile(r'CA_ROTOR(\d+)_AZ\s+(-?[\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE),
        'km': re.compile(r'CA_ROTOR(\d+)_KM\s+(-?[\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE),
        'ct': re.compile(r'CA_ROTOR(\d+)_CT\s+(-?[\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE),
    }

    # Pattern to detect rotor count
    ROTOR_COUNT_PATTERN = re.compile(r'CA_ROTOR_COUNT\s+(\d+)', re.IGNORECASE)

    # Pattern to detect if file is a PX4 airframe
    DETECTION_PATTERN = re.compile(r'CA_ROTOR\d+_[PAK]', re.IGNORECASE)

    @property
    def name(self) -> str:
        return "PX4"

    @property
    def extensions(self) -> list[str]:
        # PX4 airframes often have no extension or custom extensions
        return []

    def can_parse(self, source: Union[str, Path]) -> bool:
        """
        Check if source is a PX4 airframe file.

        Detects PX4 format by looking for CA_ROTOR parameters.
        """
        content = self._get_content(source)
        if content is None:
            return False

        # Check for CA_ROTOR parameters
        return bool(self.DETECTION_PATTERN.search(content))

    def parse(self, source: Union[str, Path]) -> ActuatorConfig:
        """
        Parse PX4 airframe file to ActuatorConfig.

        Args:
            source: File path or string content

        Returns:
            ActuatorConfig with parsed actuators
        """
        content = self._get_content(source)
        if content is None:
            raise ValueError(f"Could not read source: {source}")

        # Extract airframe name from file path or content
        name = self._extract_name(source, content)

        # Parse rotor parameters
        rotors = self._parse_rotors(content)

        if not rotors:
            raise ValueError("No CA_ROTOR parameters found in source")

        # Convert to Actuator objects
        actuators = []
        for idx, rotor_data in sorted(rotors.items()):
            actuator = Actuator(
                id=idx,
                name=f"Rotor_{idx}",
                position=(
                    rotor_data.get('px', 0.0),
                    rotor_data.get('py', 0.0),
                    rotor_data.get('pz', 0.0),
                ),
                axis=(
                    rotor_data.get('ax', 0.0),
                    rotor_data.get('ay', 0.0),
                    rotor_data.get('az', 1.0),
                ),
                coefficient=rotor_data.get('ct', 1.0),
                moment_ratio=rotor_data.get('km', 0.0),
            )
            actuators.append(actuator)

        return ActuatorConfig(
            name=name,
            actuators=actuators,
            frame="NED",  # PX4 uses NED frame
            units="meters",
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
            # Assume it's content string
            return source

        return None

    def _extract_name(self, source: Union[str, Path], content: str) -> str:
        """Extract configuration name from source."""
        # Try to get from file path
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                return f"PX4: {path.stem}"

        # Try to find airframe name in content (often in comments)
        # Look for patterns like: # Airframe: MyQuad
        name_match = re.search(r'#\s*(?:Airframe|Name|Vehicle):\s*(.+)', content, re.IGNORECASE)
        if name_match:
            return name_match.group(1).strip()

        # Look for SYS_AUTOSTART comment
        autostart_match = re.search(r'SYS_AUTOSTART\s+(\d+)', content)
        if autostart_match:
            return f"PX4 Airframe {autostart_match.group(1)}"

        return "PX4 Airframe"

    def _parse_rotors(self, content: str) -> dict[int, dict]:
        """Parse all rotor parameters from content."""
        rotors: dict[int, dict] = {}

        for param, pattern in self.PARAM_PATTERNS.items():
            for match in pattern.finditer(content):
                rotor_idx = int(match.group(1))
                value = float(match.group(2))

                if rotor_idx not in rotors:
                    rotors[rotor_idx] = {'ct': 1.0, 'km': 0.0}

                rotors[rotor_idx][param] = value

        return rotors


def parse_px4_airframe(path: Union[str, Path]) -> ActuatorConfig:
    """
    Parse a PX4 airframe file.

    Convenience function for quick parsing.

    Args:
        path: Path to PX4 airframe file

    Returns:
        ActuatorConfig object
    """
    parser = PX4Parser()
    return parser.parse(path)


def generate_px4_params(config: ActuatorConfig, start_index: int = 0) -> str:
    """
    Generate PX4 airframe parameter strings from config.

    Args:
        config: ActuatorConfig to export
        start_index: Starting rotor index (default 0)

    Returns:
        String with param set-default commands
    """
    lines = []
    lines.append("# Control Allocation Rotor Parameters")
    lines.append(f"param set-default CA_ROTOR_COUNT {config.n_actuators}")
    lines.append("")

    for actuator in config.actuators:
        idx = start_index + actuator.id
        lines.append(f"# Rotor {idx}: {actuator.name}")

        # Position
        lines.append(f"param set-default CA_ROTOR{idx}_PX {actuator.position[0]:.5f}")
        lines.append(f"param set-default CA_ROTOR{idx}_PY {actuator.position[1]:.5f}")
        lines.append(f"param set-default CA_ROTOR{idx}_PZ {actuator.position[2]:.5f}")

        # Thrust axis
        lines.append(f"param set-default CA_ROTOR{idx}_AX {actuator.axis[0]:.5f}")
        lines.append(f"param set-default CA_ROTOR{idx}_AY {actuator.axis[1]:.5f}")
        lines.append(f"param set-default CA_ROTOR{idx}_AZ {actuator.axis[2]:.5f}")

        # Coefficients
        lines.append(f"param set-default CA_ROTOR{idx}_CT {actuator.coefficient}")
        lines.append(f"param set-default CA_ROTOR{idx}_KM {actuator.moment_ratio}")

        lines.append("")

    return "\n".join(lines)
