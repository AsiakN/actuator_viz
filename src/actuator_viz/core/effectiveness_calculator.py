#!/usr/bin/env python3
"""
PX4 Control Allocation Effectiveness Matrix Calculator

Computes the 6×N effectiveness matrix from rotor coordinates,
analyzes controllability, detects configuration issues, and
generates PX4 parameters.

Usage:
    python effectiveness_calculator.py                     # Use built-in UUV config
    python effectiveness_calculator.py --airframe <file>   # Parse airframe file

Author: Generated for UUV Reconbot project
"""

import numpy as np
import argparse
import re
from typing import Optional

# Visualization module (optional)
try:
    from src.visualizers.effectiveness_visualizer import (
        generate_visualization_report,
        check_plotly_available
    )
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False


def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cross product a × b.

    Cross product formula:
        (a × b)_x = a_y * b_z - a_z * b_y
        (a × b)_y = a_z * b_x - a_x * b_z
        (a × b)_z = a_x * b_y - a_y * b_x

    Args:
        a: 3D vector [x, y, z]
        b: 3D vector [x, y, z]

    Returns:
        3D vector perpendicular to both a and b
    """
    return np.array([
        a[1] * b[2] - a[2] * b[1],  # Roll component
        a[2] * b[0] - a[0] * b[2],  # Pitch component
        a[0] * b[1] - a[1] * b[0],  # Yaw component
    ])


def compute_effectiveness_matrix(rotors: list[dict]) -> np.ndarray:
    """
    Compute 6×N effectiveness matrix from rotor parameters.

    The effectiveness matrix maps motor commands to forces and torques:
        [roll, pitch, yaw, Fx, Fy, Fz]^T = E × [m0, m1, ..., mN]^T

    For each rotor:
        thrust = CT × axis
        torque = CT × (position × axis) - CT × KM × axis

    Args:
        rotors: List of rotor dicts, each containing:
            px, py, pz: position relative to COG (meters)
            ax, ay, az: thrust axis direction (normalized)
            ct: thrust coefficient (default 1.0)
            km: moment ratio for yaw from spin (default 0.0)

    Returns:
        6×N numpy array where:
            Rows 0-2: Roll, Pitch, Yaw torque contributions
            Rows 3-5: Fx, Fy, Fz thrust contributions
    """
    n_rotors = len(rotors)
    effectiveness = np.zeros((6, n_rotors))

    for i, rotor in enumerate(rotors):
        # Extract parameters with defaults
        position = np.array([
            rotor.get('px', 0.0),
            rotor.get('py', 0.0),
            rotor.get('pz', 0.0),
        ])

        axis = np.array([
            rotor.get('ax', 0.0),
            rotor.get('ay', 0.0),
            rotor.get('az', 0.0),
        ])

        ct = rotor.get('ct', 1.0)  # Thrust coefficient
        km = rotor.get('km', 0.0)  # Moment ratio

        # Normalize axis if not already normalized
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0:
            axis = axis / axis_norm

        # Compute thrust contribution
        thrust = ct * axis

        # Compute torque contribution
        # torque = CT × (position × axis) - CT × KM × axis
        moment_from_position = ct * cross_product(position, axis)
        moment_from_spin = ct * km * axis
        torque = moment_from_position - moment_from_spin

        # Fill effectiveness matrix
        effectiveness[0, i] = torque[0]  # Roll
        effectiveness[1, i] = torque[1]  # Pitch
        effectiveness[2, i] = torque[2]  # Yaw
        effectiveness[3, i] = thrust[0]  # Fx
        effectiveness[4, i] = thrust[1]  # Fy
        effectiveness[5, i] = thrust[2]  # Fz

    return effectiveness


def analyze_controllability(effectiveness: np.ndarray) -> dict:
    """
    Analyze controllability of the system.

    A system is fully controllable if the effectiveness matrix has rank 6
    (can independently control all 6 DOF: roll, pitch, yaw, x, y, z).

    Args:
        effectiveness: 6×N effectiveness matrix

    Returns:
        Dictionary with:
            rank: Matrix rank (should be 6 for full controllability)
            controllable: True if rank == 6
            singular_values: SVD singular values (indicates control authority)
            weak_axes: List of axes with low control authority
            condition_number: Ratio of max/min singular values
    """
    # Compute SVD
    U, S, Vt = np.linalg.svd(effectiveness)

    # Rank is number of non-zero singular values
    tolerance = max(effectiveness.shape) * np.finfo(float).eps * S[0]
    rank = np.sum(S > tolerance)

    # Identify weak axes (low singular values)
    axis_names = ['Roll', 'Pitch', 'Yaw', 'Thrust X', 'Thrust Y', 'Thrust Z']
    weak_axes = []
    threshold = 0.1 * S[0]  # Less than 10% of max singular value

    # Map singular values to control axes via U matrix
    for i, (sv, axis) in enumerate(zip(S, axis_names)):
        if sv < threshold:
            weak_axes.append(f"{axis} (σ={sv:.4f})")

    # Condition number (lower is better, high means ill-conditioned)
    condition_number = S[0] / S[-1] if S[-1] > tolerance else float('inf')

    return {
        'rank': rank,
        'controllable': rank == 6,
        'singular_values': S,
        'weak_axes': weak_axes,
        'condition_number': condition_number,
        'U': U,  # Left singular vectors (axis directions)
    }


def print_effectiveness_matrix(effectiveness: np.ndarray,
                                rotor_names: Optional[list[str]] = None):
    """
    Pretty-print the effectiveness matrix.

    Args:
        effectiveness: 6×N effectiveness matrix
        rotor_names: Optional list of rotor names
    """
    n_rotors = effectiveness.shape[1]

    if rotor_names is None:
        rotor_names = [f"Rotor{i}" for i in range(n_rotors)]

    axis_names = ['Roll τ', 'Pitch τ', 'Yaw τ', 'Thrust X', 'Thrust Y', 'Thrust Z']

    # Header
    print("\n" + "=" * 80)
    print("EFFECTIVENESS MATRIX (6 × N)")
    print("=" * 80)

    # Column headers
    header = f"{'Axis':<12}"
    for name in rotor_names:
        header += f"{name:>10}"
    print(header)
    print("-" * 80)

    # Matrix rows
    for i, axis in enumerate(axis_names):
        row = f"{axis:<12}"
        for j in range(n_rotors):
            val = effectiveness[i, j]
            if abs(val) < 1e-6:
                row += f"{'0':>10}"
            else:
                row += f"{val:>10.4f}"
        print(row)

    print("=" * 80)


def detect_issues(effectiveness: np.ndarray, rotors: list[dict]) -> list[str]:
    """
    Detect common configuration issues.

    Checks for:
    - Rotors with same position and thrust direction (redundant)
    - Missing DOF control (rank < 6)
    - Rotors that can't contribute to yaw
    - Asymmetric configurations

    Args:
        effectiveness: 6×N effectiveness matrix
        rotors: List of rotor configurations

    Returns:
        List of issue descriptions
    """
    issues = []
    n_rotors = len(rotors)

    # Check rank / controllability
    result = analyze_controllability(effectiveness)
    if not result['controllable']:
        issues.append(f"NOT FULLY CONTROLLABLE: Rank is {result['rank']}, need 6")
        if result['weak_axes']:
            issues.append(f"  Weak/missing axes: {', '.join(result['weak_axes'])}")

    # Check for rotors with same yaw contribution sign AND similar position
    # (actually redundant rotors that can't provide bidirectional control)
    yaw_row = effectiveness[2, :]
    for i in range(n_rotors):
        for j in range(i + 1, n_rotors):
            px_i, px_j = rotors[i].get('px', 0), rotors[j].get('px', 0)
            py_i, py_j = rotors[i].get('py', 0), rotors[j].get('py', 0)
            yaw_i, yaw_j = yaw_row[i], yaw_row[j]

            # Both at similar X AND Y position with same-sign yaw contribution
            same_x = abs(px_i - px_j) < 0.05
            same_y = abs(py_i - py_j) < 0.05
            same_yaw_sign = (yaw_i > 0.01 and yaw_j > 0.01) or (yaw_i < -0.01 and yaw_j < -0.01)

            if same_x and same_y and same_yaw_sign:
                issues.append(
                    f"Rotor {i} and {j}: Similar position (X:{px_i:.2f}/{px_j:.2f}, Y:{py_i:.2f}/{py_j:.2f}) "
                    f"with same yaw sign ({yaw_i:.3f}/{yaw_j:.3f}) - redundant"
                )

    # Check for zero yaw authority
    yaw_row = effectiveness[2, :]
    if np.allclose(yaw_row, 0, atol=1e-6):
        issues.append("NO YAW CONTROL: All rotors have zero yaw contribution")
    elif np.all(yaw_row >= 0) or np.all(yaw_row <= 0):
        direction = "positive" if np.all(yaw_row >= 0) else "negative"
        issues.append(f"UNIDIRECTIONAL YAW: All rotors produce {direction} yaw only")

    # Check for zero thrust in each axis
    for axis_idx, axis_name in [(3, 'X'), (4, 'Y'), (5, 'Z')]:
        thrust_row = effectiveness[axis_idx, :]
        if np.allclose(thrust_row, 0, atol=1e-6):
            issues.append(f"NO {axis_name} THRUST: No rotors contribute to {axis_name}-axis thrust")

    # Check condition number
    if result['condition_number'] > 100:
        issues.append(
            f"HIGH CONDITION NUMBER ({result['condition_number']:.1f}): "
            "Control authority is very uneven across axes"
        )

    return issues


def generate_px4_params(rotors: list[dict], start_index: int = 0) -> str:
    """
    Generate PX4 airframe parameter strings.

    Args:
        rotors: List of rotor configurations
        start_index: Starting rotor index (default 0)

    Returns:
        String with param set-default commands
    """
    lines = []
    lines.append("# Control Allocation Rotor Parameters")
    lines.append(f"param set-default CA_ROTOR_COUNT {len(rotors)}")
    lines.append("")

    for i, rotor in enumerate(rotors):
        idx = start_index + i
        lines.append(f"# Rotor {idx}")

        # Position
        lines.append(f"param set-default CA_ROTOR{idx}_PX {rotor.get('px', 0.0):.5f}")
        lines.append(f"param set-default CA_ROTOR{idx}_PY {rotor.get('py', 0.0):.5f}")
        lines.append(f"param set-default CA_ROTOR{idx}_PZ {rotor.get('pz', 0.0):.5f}")

        # Thrust axis
        lines.append(f"param set-default CA_ROTOR{idx}_AX {rotor.get('ax', 0.0):.5f}")
        lines.append(f"param set-default CA_ROTOR{idx}_AY {rotor.get('ay', 0.0):.5f}")
        lines.append(f"param set-default CA_ROTOR{idx}_AZ {rotor.get('az', 0.0):.5f}")

        # Moment ratio
        lines.append(f"param set-default CA_ROTOR{idx}_KM {rotor.get('km', 0.0)}")

        lines.append("")

    return "\n".join(lines)


def parse_airframe_file(filepath: str) -> list[dict]:
    """
    Parse PX4 airframe file to extract rotor configurations.

    Args:
        filepath: Path to airframe file

    Returns:
        List of rotor dicts
    """
    rotors = {}

    # Patterns for CA_ROTOR parameters
    patterns = {
        'px': re.compile(r'CA_ROTOR(\d+)_PX\s+(-?[\d.]+)'),
        'py': re.compile(r'CA_ROTOR(\d+)_PY\s+(-?[\d.]+)'),
        'pz': re.compile(r'CA_ROTOR(\d+)_PZ\s+(-?[\d.]+)'),
        'ax': re.compile(r'CA_ROTOR(\d+)_AX\s+(-?[\d.]+)'),
        'ay': re.compile(r'CA_ROTOR(\d+)_AY\s+(-?[\d.]+)'),
        'az': re.compile(r'CA_ROTOR(\d+)_AZ\s+(-?[\d.]+)'),
        'km': re.compile(r'CA_ROTOR(\d+)_KM\s+(-?[\d.]+)'),
        'ct': re.compile(r'CA_ROTOR(\d+)_CT\s+(-?[\d.]+)'),
    }

    with open(filepath, 'r') as f:
        content = f.read()

    for param, pattern in patterns.items():
        for match in pattern.finditer(content):
            rotor_idx = int(match.group(1))
            value = float(match.group(2))

            if rotor_idx not in rotors:
                rotors[rotor_idx] = {'ct': 1.0, 'km': 0.0}

            rotors[rotor_idx][param] = value

    # Convert to sorted list
    max_idx = max(rotors.keys()) if rotors else -1
    result = []
    for i in range(max_idx + 1):
        if i in rotors:
            result.append(rotors[i])
        else:
            result.append({'px': 0, 'py': 0, 'pz': 0, 'ax': 0, 'ay': 0, 'az': 1, 'ct': 1.0, 'km': 0})

    return result


def get_uuv_reconbot_config() -> list[dict]:
    """
    Return the UUV Reconbot rotor configuration.

    8 thrusters:
    - Rotors 0-3: 45° angled (heave + sway)
    - Rotors 4-5: Forward thrusters (surge)
    - Rotors 6-7: Lateral thrusters (sway + yaw)
    """
    return [
        # Rotor 0: Bow starboard, 45° angle
        {"px": -0.42448, "py": 0.1339, "pz": -0.1167,
         "ax": 0, "ay": -0.70710678, "az": 0.70710678, "km": 0},

        # Rotor 1: Bow port, 45° angle
        {"px": -0.42448, "py": -0.13405, "pz": -0.1167,
         "ax": 0, "ay": 0.70710678, "az": 0.70710678, "km": 0},

        # Rotor 2: Stern port, 45° angle
        {"px": 0.46152, "py": -0.13405, "pz": -0.1167,
         "ax": 0, "ay": 0.70710678, "az": -0.70710678, "km": 0},

        # Rotor 3: Stern starboard, 45° angle
        {"px": 0.46152, "py": 0.13390, "pz": -0.1167,
         "ax": 0, "ay": -0.70710678, "az": -0.70710678, "km": 0},

        # Rotor 4: Bow port, forward thrust
        {"px": -0.16905, "py": -0.13930, "pz": -0.09882,
         "ax": 1, "ay": 0, "az": 0, "km": 0},

        # Rotor 5: Bow starboard, forward thrust
        {"px": -0.16905, "py": 0.13870, "pz": 0.09882,
         "ax": 1, "ay": 0, "az": 0, "km": 0},

        # Rotor 6: Stern starboard, lateral thrust
        {"px": 0.30609, "py": 0.08809, "pz": 0.15977,
         "ax": 0, "ay": 1, "az": 0, "km": 0},

        # Rotor 7: Stern starboard (BUG: should be port!)
        {"px": 0.2701, "py": 0.08809, "pz": 0.15977,
         "ax": 0, "ay": 1, "az": 0, "km": 0},
    ]


def get_uuv_reconbot_fixed() -> list[dict]:
    """
    Return the FIXED UUV Reconbot rotor configuration.

    Fixes:
    - Rotor 7: Changed to port side (PY negative) with opposite thrust
    - Rotor 5: Fixed PZ to match Rotor 4
    """
    config = get_uuv_reconbot_config()

    # Fix Rotor 5: Same Z as Rotor 4
    config[5]["pz"] = -0.09882

    # Fix Rotor 7: Port side with opposite thrust
    config[7]["py"] = -0.08809
    config[7]["ay"] = -1  # Thrust left instead of right

    return config


def main():
    parser = argparse.ArgumentParser(
        description='PX4 Control Allocation Effectiveness Matrix Calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze built-in UUV configuration
    python effectiveness_calculator.py

    # Analyze with fixed configuration
    python effectiveness_calculator.py --fixed

    # Parse and analyze an airframe file
    python effectiveness_calculator.py --airframe path/to/airframe

    # Generate PX4 parameters
    python effectiveness_calculator.py --generate-params

    # Generate interactive HTML visualization
    python effectiveness_calculator.py --visualize

    # Fixed config with custom output path
    python effectiveness_calculator.py --fixed --visualize --output report.html
        """
    )

    parser.add_argument('--airframe', type=str, help='Path to PX4 airframe file to parse')
    parser.add_argument('--fixed', action='store_true', help='Use fixed UUV configuration')
    parser.add_argument('--generate-params', action='store_true', help='Generate PX4 parameters')
    parser.add_argument('--quiet', action='store_true', help='Only show issues')
    parser.add_argument('--visualize', action='store_true', help='Generate interactive HTML visualization')
    parser.add_argument('--output', type=str, default='effectiveness_report.html',
                        help='Output path for visualization (default: effectiveness_report.html)')

    args = parser.parse_args()

    # Get rotor configuration
    if args.airframe:
        print(f"Parsing airframe file: {args.airframe}")
        rotors = parse_airframe_file(args.airframe)
    elif args.fixed:
        print("Using FIXED UUV Reconbot configuration")
        rotors = get_uuv_reconbot_fixed()
    else:
        print("Using current UUV Reconbot configuration")
        rotors = get_uuv_reconbot_config()

    print(f"Found {len(rotors)} rotors\n")

    # Compute effectiveness matrix
    E = compute_effectiveness_matrix(rotors)

    # Print matrix
    if not args.quiet:
        print_effectiveness_matrix(E)

    # Analyze controllability
    result = analyze_controllability(E)

    print("\n" + "=" * 80)
    print("CONTROLLABILITY ANALYSIS")
    print("=" * 80)
    print(f"Matrix Rank: {result['rank']} (need 6 for full control)")
    print(f"Controllable: {'YES' if result['controllable'] else 'NO'}")
    print(f"Condition Number: {result['condition_number']:.2f}")
    print(f"\nSingular Values:")
    axis_names = ['Roll', 'Pitch', 'Yaw', 'Fx', 'Fy', 'Fz']
    for i, (sv, axis) in enumerate(zip(result['singular_values'], axis_names)):
        bar = '█' * int(sv / result['singular_values'][0] * 30)
        print(f"  {axis:>10}: {sv:8.4f} {bar}")

    # Detect issues
    issues = detect_issues(E, rotors)

    print("\n" + "=" * 80)
    print("ISSUE DETECTION")
    print("=" * 80)
    if issues:
        for issue in issues:
            print(f"⚠️  {issue}")
    else:
        print("✅ No issues detected")

    # Generate parameters if requested
    if args.generate_params:
        print("\n" + "=" * 80)
        print("PX4 PARAMETERS")
        print("=" * 80)
        print(generate_px4_params(rotors))

    # Generate visualization if requested
    if args.visualize:
        print("\n" + "=" * 80)
        print("VISUALIZATION")
        print("=" * 80)

        if not VISUALIZER_AVAILABLE:
            print("ERROR: Visualization module not available.")
            print("Ensure effectiveness_visualizer.py is in the same directory.")
        elif not check_plotly_available():
            print("ERROR: Plotly is not installed.")
            print("Install with: pip install plotly>=5.0.0")
        else:
            config_name = "Fixed UUV" if args.fixed else ("Airframe" if args.airframe else "UUV Reconbot")
            output_path = generate_visualization_report(
                rotors=rotors,
                effectiveness=E,
                controllability_result=result,
                issues=issues,
                output_path=args.output,
                title=f'{config_name} Thruster Configuration'
            )
            print(f"Generated: {output_path}")

    print()


if __name__ == "__main__":
    main()