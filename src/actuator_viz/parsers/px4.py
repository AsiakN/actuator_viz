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
import warnings
from pathlib import Path

from ..core.models import Actuator, ActuatorConfig, CoordinateFrame
from .base import NUMBER_RE, ConfigParser


class PX4Parser(ConfigParser):
    """
    Parser for PX4 airframe configuration files.

    Extracts CA_ROTOR parameters to build actuator configurations.
    Supports both `param set-default` and `param set` syntax.
    """

    # Regex patterns for CA_ROTOR parameters
    # Matches: CA_ROTOR0_PX, CA_ROTOR10_AZ, etc. The value uses the shared
    # NUMBER_RE so signs and scientific notation parse and malformed tokens
    # (e.g. a bare ".") are never captured.
    PARAM_PATTERNS = {
        key: re.compile(rf"CA_ROTOR(\d+)_{key.upper()}\s+({NUMBER_RE})", re.IGNORECASE)
        for key in ("px", "py", "pz", "ax", "ay", "az", "km", "ct")
    }

    # Which per-rotor keys carry actual layout geometry (vs. coefficients).
    _GEOMETRY_KEYS = frozenset({"px", "py", "pz", "ax", "ay", "az"})

    # Pattern to detect rotor count
    ROTOR_COUNT_PATTERN = re.compile(r"CA_ROTOR_COUNT\s+(\d+)", re.IGNORECASE)

    # Pattern to detect if file is a PX4 airframe
    DETECTION_PATTERN = re.compile(r"CA_ROTOR\d+_[PAK]", re.IGNORECASE)

    @property
    def name(self) -> str:
        return "PX4"

    @property
    def extensions(self) -> list[str]:
        # PX4 airframes often have no extension or custom extensions
        return []

    def can_parse(self, source: str | Path) -> bool:
        """
        Check if source is a PX4 airframe file.

        Detects PX4 format by looking for CA_ROTOR parameters.
        """
        content = self._get_content(source)
        if content is None:
            return False

        # Check for CA_ROTOR parameters
        return bool(self.DETECTION_PATTERN.search(content))

    def parse(self, source: str | Path) -> ActuatorConfig:
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
            raise ValueError(
                "No CA_ROTOR parameters found. This looks like a PX4 file but "
                "carries no control-allocation rotor definitions."
            )

        # Stock PX4 airframes frequently set only SYS_AUTOSTART / CA_ROTOR_COUNT
        # and inherit the actual geometry from defaults — there's nothing to
        # analyze. Require at least one position/axis value to be present.
        if not any(self._GEOMETRY_KEYS & r.keys() for r in rotors.values()):
            raise ValueError(
                "Found CA_ROTOR references but no rotor geometry "
                "(CA_ROTOR*_PX/PY/PZ/AX/AY/AZ). Stock airframes often inherit "
                "geometry from PX4 defaults rather than listing it; export the "
                "resolved parameters or describe the layout in a YAML config."
            )

        # Sanity-check against a declared rotor count, if present.
        count_match = self.ROTOR_COUNT_PATTERN.search(content)
        if count_match:
            declared = int(count_match.group(1))
            if declared != len(rotors):
                warnings.warn(
                    f"CA_ROTOR_COUNT is {declared} but geometry was found for "
                    f"{len(rotors)} rotor(s); analyzing the {len(rotors)} defined.",
                    stacklevel=2,
                )

        # Convert to Actuator objects
        actuators = []
        for idx, rotor_data in sorted(rotors.items()):
            axis = (
                rotor_data.get("ax", 0.0),
                rotor_data.get("ay", 0.0),
                rotor_data.get("az", 1.0),
            )
            if axis == (0.0, 0.0, 0.0):
                raise ValueError(
                    f"Rotor {idx} has a zero thrust axis "
                    f"(CA_ROTOR{idx}_AX/AY/AZ all 0); it has no thrust direction."
                )
            actuator = Actuator(
                id=idx,
                name=f"Rotor_{idx}",
                position=(
                    rotor_data.get("px", 0.0),
                    rotor_data.get("py", 0.0),
                    rotor_data.get("pz", 0.0),
                ),
                axis=axis,
                coefficient=rotor_data.get("ct", 1.0),
                moment_ratio=rotor_data.get("km", 0.0),
            )
            actuators.append(actuator)

        return ActuatorConfig(
            name=name,
            actuators=actuators,
            frame=CoordinateFrame.NED,  # PX4 uses NED frame
            units="meters",
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
            # Assume it's content string
            return source

        return None

    def _extract_name(self, source: str | Path, content: str) -> str:
        """Extract configuration name from source."""
        # Try to get from file path
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                return f"PX4: {path.stem}"

        # Try to find airframe name in content (often in comments)
        # Look for patterns like: # Airframe: MyQuad
        name_match = re.search(r"#\s*(?:Airframe|Name|Vehicle):\s*(.+)", content, re.IGNORECASE)
        if name_match:
            return name_match.group(1).strip()

        # Look for SYS_AUTOSTART comment
        autostart_match = re.search(r"SYS_AUTOSTART\s+(\d+)", content)
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
                    rotors[rotor_idx] = {"ct": 1.0, "km": 0.0}

                rotors[rotor_idx][param] = value

        return rotors


def parse_px4_airframe(path: str | Path) -> ActuatorConfig:
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
