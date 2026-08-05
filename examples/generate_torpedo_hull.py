#!/usr/bin/env python3
"""
Generate a realistic torpedo-AUV hull as a watertight OBJ mesh.

The hull is a surface of revolution about the +X (forward) axis:

    elliptical ogive nose  ->  cylindrical mid-body  ->  tapered tail

Both ends close to a point so the mesh is watertight. Dimensions are chosen so
the actuators declared in ``torpedo_auv.yaml`` sit on (or just outside) the hull
surface: the bow canted thrusters land at the body/nose junction, the aft surge
thrusters ride the tapered tail.

Run from the repo root to regenerate the example mesh::

    python examples/generate_torpedo_hull.py

Frame: +X forward, +Y port, +Z up. Units: meters. Centered on the CoG (origin)
along Y/Z; the X origin matches the actuator frame in torpedo_auv.yaml.
"""

from __future__ import annotations

import math
from pathlib import Path

# --- Hull dimensions (meters) ------------------------------------------------
RADIUS = 0.13  # body radius; diameter 0.26 m encloses the ±0.12 m thruster ring

NOSE_TIP_X = 0.42  # forward-most point
NOSE_BASE_X = 0.30  # nose/body junction (bow canted thrusters sit here)
TAIL_BASE_X = -0.35  # body/tail junction (stern canted thrusters sit here)
TAIL_TIP_X = -0.52  # aft-most point

N_THETA = 24  # segments around the circumference
N_NOSE = 10  # ring stations along the nose (excludes the tip apex)
N_BODY = 5  # ring stations along the cylindrical body
N_TAIL = 10  # ring stations along the tail (excludes the tip apex)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _profile_radius(x: float) -> float:
    """Hull radius at station ``x`` — the meridian that gets revolved."""
    if x >= NOSE_BASE_X:
        # Elliptical ogive nose: r = R * sqrt(1 - ((x - base)/len)^2)
        frac = (x - NOSE_BASE_X) / (NOSE_TIP_X - NOSE_BASE_X)
        return RADIUS * math.sqrt(max(0.0, 1.0 - frac * frac))
    if x <= TAIL_BASE_X:
        # Elliptical tail taper, gentler than the nose (longer run).
        frac = (TAIL_BASE_X - x) / (TAIL_BASE_X - TAIL_TIP_X)
        return RADIUS * math.sqrt(max(0.0, 1.0 - frac * frac))
    return RADIUS  # cylindrical mid-body


def _station_xs() -> list[float]:
    """Ordered X stations from tail tip to nose tip, apex points excluded."""
    xs: list[float] = []
    # Tail: TAIL_TIP_X (apex, skipped) -> TAIL_BASE_X
    for i in range(1, N_TAIL + 1):
        xs.append(_lerp(TAIL_TIP_X, TAIL_BASE_X, i / N_TAIL))
    # Body: strictly between the junctions (endpoints already added).
    for i in range(1, N_BODY):
        xs.append(_lerp(TAIL_BASE_X, NOSE_BASE_X, i / N_BODY))
    # Nose: NOSE_BASE_X -> NOSE_TIP_X (apex, skipped)
    for i in range(N_NOSE):
        xs.append(_lerp(NOSE_BASE_X, NOSE_TIP_X, i / N_NOSE))
    # Deduplicate shared junctions while preserving order.
    out: list[float] = []
    for x in xs:
        if not out or abs(x - out[-1]) > 1e-9:
            out.append(x)
    return out


def build_hull() -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Return (vertices, faces) for a watertight revolved torpedo hull."""
    xs = _station_xs()

    vertices: list[tuple[float, float, float]] = []
    # Ring vertices: ring r, angle t -> vertex index (1 + r * N_THETA + t) later.
    for x in xs:
        r = _profile_radius(x)
        for t in range(N_THETA):
            ang = 2.0 * math.pi * t / N_THETA
            vertices.append((x, r * math.cos(ang), r * math.sin(ang)))

    tail_apex = (TAIL_TIP_X, 0.0, 0.0)
    nose_apex = (NOSE_TIP_X, 0.0, 0.0)
    vertices.append(tail_apex)
    vertices.append(nose_apex)

    n_rings = len(xs)
    tail_apex_idx = n_rings * N_THETA  # 0-based
    nose_apex_idx = tail_apex_idx + 1

    def ring_vertex(ring: int, t: int) -> int:
        return ring * N_THETA + (t % N_THETA)

    faces: list[tuple[int, int, int]] = []

    # Tail apex fan (outward winding: apex is at -X, so wind CW seen from -X).
    for t in range(N_THETA):
        a = ring_vertex(0, t)
        b = ring_vertex(0, t + 1)
        faces.append((tail_apex_idx, b, a))

    # Quads between consecutive rings, split into two triangles.
    for ring in range(n_rings - 1):
        for t in range(N_THETA):
            a = ring_vertex(ring, t)
            b = ring_vertex(ring, t + 1)
            c = ring_vertex(ring + 1, t + 1)
            d = ring_vertex(ring + 1, t)
            faces.append((a, b, c))
            faces.append((a, c, d))

    # Nose apex fan.
    last = n_rings - 1
    for t in range(N_THETA):
        a = ring_vertex(last, t)
        b = ring_vertex(last, t + 1)
        faces.append((nose_apex_idx, a, b))

    return vertices, faces


def write_obj(path: Path) -> None:
    vertices, faces = build_hull()
    lines = [
        "# Torpedo-AUV hull: elliptical ogive nose + cylindrical body + tapered tail.",
        "# Watertight surface of revolution about +X. Authored in meters, centered on",
        "# the vehicle CoG (origin). Referenced by torpedo_auv.yaml.",
        "# Regenerate with: python examples/generate_torpedo_hull.py",
        f"# {len(vertices)} vertices, {len(faces)} faces.",
    ]
    for x, y, z in vertices:
        lines.append(f"v {x:.5f} {y:.5f} {z:.5f}")
    for i, j, k in faces:  # OBJ face indices are 1-based.
        lines.append(f"f {i + 1} {j + 1} {k + 1}")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}: {len(vertices)} vertices, {len(faces)} faces")


if __name__ == "__main__":
    write_obj(Path(__file__).resolve().parent / "torpedo_auv.obj")
