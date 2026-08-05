#!/usr/bin/env python3
"""
Generate a realistic quadcopter-frame mesh (OBJ) for vehicle rendering.

Builds an X-configuration quad as a triangle-soup assembly — central body,
four diagonal arms, four motor cans, and four two-blade propellers — rather than
a single watertight solid (watertightness doesn't matter for Plotly Mesh3d, and
separate parts read as a real airframe). Geometry is centered on the CoG
(origin) and sized to match the motor positions in ``quadcopter.yaml``.

Run from the repo root to regenerate the example mesh::

    python examples/generate_quadcopter_mesh.py

The module also exposes ``build_body()`` and ``build_props(angle)`` so a renderer
can spin the propellers frame-to-frame (see the README GIF).

Frame: +X forward, +Y left, +Z up. Units: meters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --- Frame dimensions (meters) ----------------------------------------------
ARM_REACH = 0.15  # motor offset along each of X and Y (matches quadcopter.yaml)
MOTORS = {
    0: (ARM_REACH, -ARM_REACH),   # front-right
    1: (-ARM_REACH, ARM_REACH),   # back-left
    2: (ARM_REACH, ARM_REACH),    # front-left
    3: (-ARM_REACH, -ARM_REACH),  # back-right
}

PROP_RADIUS = 0.075  # ~6-inch propellers
PROP_Z = 0.034       # propeller disc height above the frame plane


# --- Primitive builders: each returns (vertices Nx3, faces Mx3) --------------
def _box(center, size) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    v = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ], dtype=float)
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=int)
    return v, f


def _cylinder(center, radius, height, seg=18) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    ang = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    ring_x, ring_y = cx + radius * np.cos(ang), cy + radius * np.sin(ang)
    bottom = np.stack([ring_x, ring_y, np.full(seg, cz)], axis=1)
    top = np.stack([ring_x, ring_y, np.full(seg, cz + height)], axis=1)
    v = np.concatenate([bottom, top, [[cx, cy, cz]], [[cx, cy, cz + height]]])
    cb, ct = 2 * seg, 2 * seg + 1
    f = []
    for i in range(seg):
        j = (i + 1) % seg
        f.append([i, j, seg + j])
        f.append([i, seg + j, seg + i])
        f.append([cb, j, i])              # bottom cap
        f.append([ct, seg + i, seg + j])  # top cap
    return v, np.array(f, dtype=int)


def _blade(motor, angle, length, width, thick, z) -> tuple[np.ndarray, np.ndarray]:
    """A single propeller blade: a thin box rotated ``angle`` about the motor."""
    hub = 0.008
    v = np.array([
        [hub, -width / 2, z - thick / 2], [length, -width / 2, z - thick / 2],
        [length, width / 2, z - thick / 2], [hub, width / 2, z - thick / 2],
        [hub, -width / 2, z + thick / 2], [length, -width / 2, z + thick / 2],
        [length, width / 2, z + thick / 2], [hub, width / 2, z + thick / 2],
    ], dtype=float)
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=int)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    v = v @ rot.T + np.array([motor[0], motor[1], 0.0])
    return v, f


def _combine(parts) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, offset = [], [], 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def build_body() -> tuple[np.ndarray, np.ndarray]:
    """Static frame: central body, canopy, four arms, four motor cans."""
    parts = [
        _box((0.0, 0.0, 0.0), (0.10, 0.10, 0.035)),      # main body
        _box((0.0, 0.0, 0.028), (0.06, 0.075, 0.028)),   # canopy / battery
    ]
    for mx, my in MOTORS.values():
        angle = np.arctan2(my, mx)
        # Arm: a thin box from the hub edge out to the motor.
        arm, arm_f = _blade((0.0, 0.0), angle, np.hypot(mx, my), 0.018, 0.012, 0.0)
        parts.append((arm, arm_f))
        parts.append(_cylinder((mx, my, -0.006), 0.019, 0.03))  # motor can
    return _combine(parts)


def build_props(angle: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Four two-blade propellers, all rotated by ``angle`` radians."""
    parts = []
    for i, (mx, my) in MOTORS.items():
        # Alternate spin direction like a real quad (CW/CCW pairs).
        a = angle if i in (0, 1) else -angle
        parts.append(_blade((mx, my), a, PROP_RADIUS, 0.016, 0.005, PROP_Z))
        parts.append(_blade((mx, my), a + np.pi, PROP_RADIUS, 0.016, 0.005, PROP_Z))
        parts.append(_cylinder((mx, my, PROP_Z - 0.004), 0.009, 0.012, seg=12))
    return _combine(parts)


def build_quad(prop_angle: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return _combine([build_body(), build_props(prop_angle)])


def write_obj(path: Path) -> None:
    vertices, faces = build_quad(prop_angle=0.0)
    lines = [
        "# Quadcopter X-frame: body + canopy + 4 arms + 4 motors + 4 two-blade props.",
        "# Triangle-soup assembly (not watertight), centered on the CoG. Meters, +Z up.",
        "# Referenced by quadcopter.yaml. Regenerate: python examples/generate_quadcopter_mesh.py",
        f"# {len(vertices)} vertices, {len(faces)} faces.",
    ]
    for x, y, z in vertices:
        lines.append(f"v {x:.5f} {y:.5f} {z:.5f}")
    for i, j, k in faces:  # OBJ indices are 1-based.
        lines.append(f"f {i + 1} {j + 1} {k + 1}")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}: {len(vertices)} vertices, {len(faces)} faces")


if __name__ == "__main__":
    write_obj(Path(__file__).resolve().parent / "quadcopter.obj")
