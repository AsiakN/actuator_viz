"""
actuator-viz: Visualize and analyze multi-actuator control allocation systems.

A generic tool for robotics developers working with ROVs, drones, hexapods,
robot arms, or any system with multiple actuators.

Quick Start:
    >>> from actuator_viz import ActuatorConfig, analyze
    >>> config = ActuatorConfig.from_yaml("my_robot.yaml")
    >>> result = analyze(config)
    >>> print(result.controllable, result.rank)
"""

__version__ = "0.1.0"

# Core data models
from .core.models import (
    Actuator,
    ActuatorConfig,
    ActuatorType,
    AnalysisResult,
    CoordinateFrame,
    Geometry,
)

# Core analysis functions
from .core.effectiveness import (
    compute_effectiveness_matrix,
    compute_effectiveness_from_rotors,
    print_effectiveness_matrix,
)

from .core.analysis import (
    analyze_controllability,
    analyze_config,
    detect_issues,
    compute_control_authority,
    compute_allocation_matrix,
    print_analysis_report,
)

# Geometry utilities
from .core.geometry import (
    cross_product,
    normalize,
    transform_vector,
)

# Parsers
from .parsers.base import (
    ConfigParser,
    ParserRegistry,
    get_registry,
    register_parser,
    parse_config,
)

from .parsers.json_yaml import (
    JsonYamlParser,
    load_yaml,
    load_json,
    parse_yaml_string,
)

from .parsers.px4 import (
    PX4Parser,
    parse_px4_airframe,
    generate_px4_params,
)

from .parsers.ardupilot import (
    ArduPilotParser,
    ArduSubFrame,
    parse_ardupilot,
    get_ardusub_frame,
    list_ardusub_frames,
)


# Convenience function
def analyze(config: ActuatorConfig) -> AnalysisResult:
    """
    Analyze an actuator configuration.

    This is the main entry point for analyzing actuator systems.

    Args:
        config: ActuatorConfig to analyze

    Returns:
        AnalysisResult with controllability analysis and detected issues

    Example:
        >>> config = ActuatorConfig.from_yaml("robot.yaml")
        >>> result = analyze(config)
        >>> if result.controllable:
        ...     print("Full 6-DOF control!")
        >>> else:
        ...     print(f"Only {result.rank} DOF controllable")
        ...     for issue in result.issues:
        ...         print(f"  - {issue}")
    """
    return analyze_config(config)


# Register default parsers (order matters for auto-detection)
register_parser(JsonYamlParser())  # Handles .yaml, .yml, .json
register_parser(PX4Parser())       # Handles PX4 airframes (CA_ROTOR params)
register_parser(ArduPilotParser()) # Handles .param, .parm files


__all__ = [
    # Version
    "__version__",
    # Models
    "Actuator",
    "ActuatorConfig",
    "ActuatorType",
    "AnalysisResult",
    "CoordinateFrame",
    "Geometry",
    # Analysis
    "analyze",
    "analyze_config",
    "analyze_controllability",
    "compute_effectiveness_matrix",
    "compute_effectiveness_from_rotors",
    "compute_control_authority",
    "compute_allocation_matrix",
    "detect_issues",
    "print_effectiveness_matrix",
    "print_analysis_report",
    # Geometry
    "cross_product",
    "normalize",
    "transform_vector",
    # Parsers
    "ConfigParser",
    "ParserRegistry",
    "JsonYamlParser",
    "get_registry",
    "register_parser",
    "parse_config",
    "load_yaml",
    "load_json",
    "parse_yaml_string",
    # PX4
    "PX4Parser",
    "parse_px4_airframe",
    "generate_px4_params",
    # ArduPilot
    "ArduPilotParser",
    "ArduSubFrame",
    "parse_ardupilot",
    "get_ardusub_frame",
    "list_ardusub_frames",
]
