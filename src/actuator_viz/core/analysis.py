"""
Controllability analysis for actuator configurations.

Provides SVD-based analysis to determine:
- Whether full 6-DOF control is achievable
- Control authority in each axis
- Configuration issues and weaknesses
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .effectiveness import compute_effectiveness_matrix, get_dof_names
from .models import ActuatorConfig, AnalysisResult


def analyze_controllability(
    effectiveness: np.ndarray,
    threshold_ratio: float = 0.1
) -> AnalysisResult:
    """
    Analyze controllability of the system using SVD.

    A system is fully controllable if the effectiveness matrix has rank 6
    (can independently control all 6 DOF: roll, pitch, yaw, x, y, z).

    The singular values indicate control authority:
    - Large values = strong control in that direction
    - Small values = weak control (hard to achieve that motion)
    - Zero values = impossible to control that DOF

    Args:
        effectiveness: 6×N effectiveness matrix
        threshold_ratio: Ratio below max singular value considered "weak"

    Returns:
        AnalysisResult with rank, controllability, singular values, etc.
    """
    # Compute SVD: E = U @ S @ Vt
    U, S, Vt = np.linalg.svd(effectiveness)

    # Determine numerical rank
    tolerance = max(effectiveness.shape) * np.finfo(float).eps * S[0]
    rank = int(np.sum(S > tolerance))

    # Identify weak axes
    dof_names = get_dof_names()
    weak_axes = []
    threshold = threshold_ratio * S[0] if S[0] > 0 else 0

    for i, sv in enumerate(S[:6]):
        if sv < threshold:
            weak_axes.append(f"{dof_names[i]} (σ={sv:.4f})")

    # Condition number (lower is better)
    # High condition number means control authority is very uneven
    if S[-1] > tolerance:
        condition_number = S[0] / S[-1]
    else:
        condition_number = float('inf')

    return AnalysisResult(
        rank=rank,
        controllable=(rank == 6),
        singular_values=S,
        condition_number=condition_number,
        weak_axes=weak_axes,
        effectiveness_matrix=effectiveness,
        U=U,
    )


def analyze_config(config: ActuatorConfig) -> AnalysisResult:
    """
    Analyze an actuator configuration.

    Convenience function that computes effectiveness matrix and analyzes it.

    Args:
        config: ActuatorConfig to analyze

    Returns:
        AnalysisResult with full analysis
    """
    effectiveness = compute_effectiveness_matrix(config)
    result = analyze_controllability(effectiveness)

    # Run issue detection
    issues = detect_issues(effectiveness, config)
    result.issues = issues

    return result


def detect_issues(
    effectiveness: np.ndarray,
    config: ActuatorConfig
) -> list[str]:
    """
    Detect common configuration issues.

    Checks for:
    - Missing DOF control (rank < 6)
    - Unidirectional control (can only rotate/translate one way)
    - Redundant actuators (same position and direction)
    - High condition number (uneven control authority)

    Args:
        effectiveness: 6×N effectiveness matrix
        config: ActuatorConfig with actuator list

    Returns:
        List of issue descriptions
    """
    issues = []
    dof_names = get_dof_names()

    # Check rank / controllability
    result = analyze_controllability(effectiveness)
    if not result.controllable:
        issues.append(f"NOT FULLY CONTROLLABLE: Rank is {result.rank}/6")
        if result.weak_axes:
            issues.append(f"  Weak/missing axes: {', '.join(result.weak_axes)}")

    # Check for unidirectional control in each DOF
    for i, dof in enumerate(dof_names):
        row = effectiveness[i, :]
        non_zero = row[np.abs(row) > 1e-6]

        if len(non_zero) == 0:
            issues.append(f"NO {dof} CONTROL: No actuators contribute to {dof}")
        elif np.all(non_zero >= 0) and len(non_zero) > 0:
            issues.append(f"UNIDIRECTIONAL {dof}: All contributions are positive only")
        elif np.all(non_zero <= 0) and len(non_zero) > 0:
            issues.append(f"UNIDIRECTIONAL {dof}: All contributions are negative only")

    # Check for redundant actuators
    n_actuators = config.n_actuators
    for i in range(n_actuators):
        for j in range(i + 1, n_actuators):
            a_i = config.actuators[i]
            a_j = config.actuators[j]

            # Check if positions are very close
            pos_diff = np.linalg.norm(
                np.array(a_i.position) - np.array(a_j.position)
            )

            # Check if axes are parallel (same or opposite direction)
            axis_dot = abs(np.dot(a_i.axis_normalized, a_j.axis_normalized))

            if pos_diff < 0.05 and axis_dot > 0.99:
                # Same position and parallel axis = potentially redundant
                # Check if they produce same effect (not opposite)
                col_i = effectiveness[:, i]
                col_j = effectiveness[:, j]
                col_diff = np.linalg.norm(col_i - col_j)

                if col_diff < 0.1:
                    issues.append(
                        f"REDUNDANT: {a_i.name} and {a_j.name} have nearly "
                        f"identical effect on the system"
                    )

    # Check condition number
    if result.condition_number > 100:
        issues.append(
            f"HIGH CONDITION NUMBER ({result.condition_number:.1f}): "
            "Control authority is very uneven across axes"
        )
    elif result.condition_number > 20:
        issues.append(
            f"MODERATE CONDITION NUMBER ({result.condition_number:.1f}): "
            "Some axes have significantly less control authority"
        )

    return issues


@dataclass
class FailureImpact:
    """
    Impact of a single actuator going offline.

    Attributes:
        actuator_id: ID of the failed actuator
        actuator_name: Name of the failed actuator
        rank: Rank of the effectiveness matrix with this actuator removed
        controllable: True if the degraded system is still fully controllable
        condition_number: Condition number of the degraded system
        lost_dofs: DOFs achievable in the full system but not after this failure
    """
    actuator_id: int
    actuator_name: str
    rank: int
    controllable: bool
    condition_number: float
    lost_dofs: list[str] = field(default_factory=list)

    @property
    def critical(self) -> bool:
        """A failure is critical if it removes control of any DOF."""
        return len(self.lost_dofs) > 0


def achievable_dofs(effectiveness: np.ndarray, tol: float = 1e-6) -> set[str]:
    """
    Determine which DOFs the system can actually command.

    A DOF is achievable if its unit wrench (e.g. pure yaw) lies in the column
    space of the effectiveness matrix. This is stronger than rank: rank gives
    the *dimension* of the controllable subspace, while this identifies *which*
    specific axes are controllable.

    Args:
        effectiveness: 6×N effectiveness matrix
        tol: Residual below which a DOF is considered achievable

    Returns:
        Set of DOF names (subset of get_dof_names()) that are controllable
    """
    dof_names = get_dof_names()
    if effectiveness.ndim != 2 or effectiveness.shape[1] == 0:
        return set()

    # Orthogonal projector onto the column space (range) of E.
    projector = effectiveness @ np.linalg.pinv(effectiveness)

    achievable = set()
    for i, name in enumerate(dof_names):
        e = np.zeros(6)
        e[i] = 1.0
        residual = np.linalg.norm(e - projector @ e)
        if residual < tol:
            achievable.add(name)
    return achievable


def simulate_failure(config: ActuatorConfig, actuator_index: int) -> FailureImpact:
    """
    Simulate a single actuator failing (going offline) and analyze the result.

    Removes the actuator's column from the effectiveness matrix and re-runs
    controllability analysis on the degraded system.

    Args:
        config: The full (healthy) actuator configuration
        actuator_index: Index into config.actuators of the failing actuator

    Returns:
        FailureImpact describing the degraded system
    """
    actuator = config.actuators[actuator_index]

    full_effectiveness = compute_effectiveness_matrix(config)
    baseline = achievable_dofs(full_effectiveness)

    reduced = np.delete(full_effectiveness, actuator_index, axis=1)

    if reduced.shape[1] == 0:
        # No actuators left — nothing is controllable.
        rank, controllable, condition = 0, False, float("inf")
        degraded: set[str] = set()
    else:
        result = analyze_controllability(reduced)
        rank = result.rank
        controllable = result.controllable
        condition = result.condition_number
        degraded = achievable_dofs(reduced)

    # DOFs we could command before, but can't with this actuator offline.
    lost = [d for d in get_dof_names() if d in baseline and d not in degraded]

    return FailureImpact(
        actuator_id=actuator.id,
        actuator_name=actuator.name,
        rank=rank,
        controllable=controllable,
        condition_number=condition,
        lost_dofs=lost,
    )


def analyze_all_failures(config: ActuatorConfig) -> list[FailureImpact]:
    """
    Simulate every single-actuator failure in turn.

    Args:
        config: The full (healthy) actuator configuration

    Returns:
        List of FailureImpact, one per actuator, in config order
    """
    return [simulate_failure(config, i) for i in range(config.n_actuators)]


def compute_control_authority(effectiveness: np.ndarray) -> dict[str, float]:
    """
    Compute control authority for each DOF.

    Control authority is the maximum achievable force/torque in each
    direction, assuming actuator outputs are bounded to [-1, 1].

    Args:
        effectiveness: 6×N effectiveness matrix

    Returns:
        Dict mapping DOF name to authority value
    """
    dof_names = get_dof_names()
    authority = {}

    for i, dof in enumerate(dof_names):
        row = effectiveness[i, :]
        # Maximum authority is sum of absolute values
        # (achieved when all actuators push in same direction)
        authority[dof] = float(np.sum(np.abs(row)))

    return authority


def compute_allocation_matrix(effectiveness: np.ndarray) -> np.ndarray:
    """
    Compute the pseudo-inverse allocation matrix.

    The allocation matrix maps desired forces/torques to actuator commands:
        [u0, u1, ..., uN]^T = A × [roll, pitch, yaw, Fx, Fy, Fz]^T

    Uses Moore-Penrose pseudo-inverse for over-determined systems.

    Args:
        effectiveness: 6×N effectiveness matrix

    Returns:
        N×6 allocation matrix
    """
    return np.linalg.pinv(effectiveness)


def print_analysis_report(result: AnalysisResult) -> None:
    """
    Print a formatted analysis report to console.

    Args:
        result: AnalysisResult from analyze_controllability
    """
    dof_names = get_dof_names()

    print("\n" + "=" * 60)
    print("CONTROLLABILITY ANALYSIS")
    print("=" * 60)

    print(f"Matrix Rank: {result.rank}/6")
    status = "YES" if result.controllable else "NO"
    print(f"Fully Controllable: {status}")
    print(f"Condition Number: {result.condition_number:.2f}")

    print("\nSingular Values (Control Authority):")
    max_sv = result.singular_values[0] if len(result.singular_values) > 0 else 1

    for i, sv in enumerate(result.singular_values[:6]):
        bar_len = int(sv / max_sv * 30) if max_sv > 0 else 0
        bar = "█" * bar_len
        dof = dof_names[i] if i < len(dof_names) else f"SV{i}"
        print(f"  {dof:>10}: {sv:8.4f} {bar}")

    if result.issues:
        print("\n" + "-" * 60)
        print("ISSUES DETECTED:")
        for issue in result.issues:
            print(f"  ⚠ {issue}")
    else:
        print("\n✓ No issues detected")

    print("=" * 60)
