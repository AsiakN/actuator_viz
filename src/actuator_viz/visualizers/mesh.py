"""
Optional STL/OBJ mesh loading for realistic vehicle rendering.

Requires the ``mesh`` extra (trimesh):  ``pip install 'actuator-viz[mesh]'``.
When trimesh is unavailable, callers fall back to a box approximation of the
vehicle body, so this stays a soft dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import trimesh

    MESH_SUPPORT = True
except ImportError:  # pragma: no cover - exercised only without the extra
    MESH_SUPPORT = False


def load_mesh(path: str | Path, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an STL/OBJ file as ``(vertices, faces)`` arrays for Plotly Mesh3d.

    Args:
        path: Path to an STL or OBJ file.
        scale: Multiplier applied to every vertex coordinate. Use it to convert
            the mesh's units into the actuator frame's meters (e.g. 0.001 for a
            mesh authored in millimeters).

    Returns:
        (vertices, faces): vertices is an (N, 3) float array, faces an (M, 3)
        int array of triangle vertex indices.

    Raises:
        ImportError: if the ``mesh`` extra (trimesh) is not installed.
        FileNotFoundError: if the file does not exist.
        ValueError: if the file cannot be read as a triangle mesh.
    """
    if not MESH_SUPPORT:
        raise ImportError(
            "Mesh rendering requires the 'mesh' extra. "
            "Install with: pip install 'actuator-viz[mesh]'"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    # force="mesh" collapses multi-part scenes into a single triangle mesh.
    loaded = trimesh.load(path, force="mesh")
    if not hasattr(loaded, "vertices") or not hasattr(loaded, "faces"):
        raise ValueError(f"Could not read a triangle mesh from {path}")

    vertices = np.asarray(loaded.vertices, dtype=float) * float(scale)
    faces = np.asarray(loaded.faces, dtype=int)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh {path} contains no geometry")
    if faces.shape[1] != 3:
        raise ValueError(f"Mesh {path} is not triangulated (got {faces.shape[1]}-sided faces)")

    return vertices, faces
