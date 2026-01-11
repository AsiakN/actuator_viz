"""
Core modules for actuator-viz.

Contains the fundamental data models, math, and analysis algorithms.
"""

from .models import (
    Actuator,
    ActuatorConfig,
    ActuatorType,
    AnalysisResult,
    CoordinateFrame,
    Geometry,
)

from .effectiveness import (
    compute_effectiveness_matrix,
    compute_effectiveness_from_rotors,
    print_effectiveness_matrix,
    get_dof_names,
)

from .analysis import (
    analyze_controllability,
    analyze_config,
    detect_issues,
    compute_control_authority,
    compute_allocation_matrix,
    print_analysis_report,
)

from .geometry import (
    cross_product,
    normalize,
    transform_vector,
    rotation_matrix_enu_to_ned,
    rotation_matrix_ned_to_enu,
)

__all__ = [
    # Models
    "Actuator",
    "ActuatorConfig",
    "ActuatorType",
    "AnalysisResult",
    "CoordinateFrame",
    "Geometry",
    # Effectiveness
    "compute_effectiveness_matrix",
    "compute_effectiveness_from_rotors",
    "print_effectiveness_matrix",
    "get_dof_names",
    # Analysis
    "analyze_controllability",
    "analyze_config",
    "detect_issues",
    "compute_control_authority",
    "compute_allocation_matrix",
    "print_analysis_report",
    # Geometry
    "cross_product",
    "normalize",
    "transform_vector",
    "rotation_matrix_enu_to_ned",
    "rotation_matrix_ned_to_enu",
]
