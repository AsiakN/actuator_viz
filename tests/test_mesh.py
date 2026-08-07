"""
Tests for STL/OBJ vehicle mesh rendering (the optional 'mesh' extra).

Mesh-loading tests are skipped when trimesh isn't installed; the box-fallback
paths are exercised unconditionally, since graceful degradation is the point.
"""

from __future__ import annotations

import numpy as np
import pytest

from actuator_viz import Actuator, ActuatorConfig, Geometry, parse_config
from actuator_viz.visualizers import create_3d_thruster_plot
from actuator_viz.visualizers.mesh import MESH_SUPPORT, load_mesh

requires_mesh = pytest.mark.skipif(not MESH_SUPPORT, reason="requires the 'mesh' extra (trimesh)")


def _rotors():
    return ActuatorConfig(
        actuators=[
            Actuator(id=0, position=(0.3, 0, 0), axis=(0, 0, 1)),
            Actuator(id=1, position=(-0.3, 0, 0), axis=(0, 0, 1)),
        ]
    ).to_rotor_list()


def _body_trace(fig):
    return next(t for t in fig.data if getattr(t, "name", None) == "Vehicle Body")


@requires_mesh
def test_load_mesh_reads_obj(examples_dir):
    vertices, faces = load_mesh(examples_dir / "torpedo_auv.obj")
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.shape[1] == 3  # triangulated
    # A real revolved hull, not the 8-vertex box approximation.
    assert len(vertices) > 100


@requires_mesh
def test_load_mesh_applies_scale(examples_dir):
    v1, _ = load_mesh(examples_dir / "torpedo_auv.obj", scale=1.0)
    v2, _ = load_mesh(examples_dir / "torpedo_auv.obj", scale=2.0)
    np.testing.assert_allclose(v2, v1 * 2.0)


@requires_mesh
def test_load_mesh_missing_file_raises(examples_dir):
    with pytest.raises(FileNotFoundError):
        load_mesh(examples_dir / "nope.obj")


@requires_mesh
def test_plot_renders_mesh_body(examples_dir):
    geo = Geometry(geometry_type="mesh", mesh_file=examples_dir / "torpedo_auv.obj")
    fig = create_3d_thruster_plot(_rotors(), geometry=geo)
    # The body trace carries the real hull's vertices, not the 8-vertex box.
    mesh_verts, _ = load_mesh(examples_dir / "torpedo_auv.obj")
    assert len(_body_trace(fig).x) == len(mesh_verts) > 8


def test_plot_without_geometry_uses_box():
    fig = create_3d_thruster_plot(_rotors())
    assert len(_body_trace(fig).x) == 8  # box fallback, backward compatible


def test_plot_missing_mesh_falls_back_to_box(examples_dir):
    # A declared-but-unloadable mesh must warn and degrade to the box, not crash.
    geo = Geometry(geometry_type="mesh", mesh_file=examples_dir / "missing.obj")
    with pytest.warns(UserWarning):
        fig = create_3d_thruster_plot(_rotors(), geometry=geo)
    assert len(_body_trace(fig).x) == 8


def test_geometry_scale_roundtrips():
    config = ActuatorConfig(
        actuators=[Actuator(id=0, position=(0, 0, 0), axis=(0, 0, 1))],
        geometry=Geometry(geometry_type="mesh", mesh_file="hull.stl", mesh_scale=0.001),
    )
    data = config.to_dict()
    assert data["geometry"]["scale"] == 0.001
    rebuilt = ActuatorConfig.from_dict(data)
    assert rebuilt.geometry.mesh_scale == 0.001
    assert str(rebuilt.geometry.mesh_file) == "hull.stl"


def test_parse_mesh_config(examples_dir):
    config = parse_config(examples_dir / "torpedo_auv.yaml")
    assert config.geometry is not None
    assert config.geometry.geometry_type == "mesh"
    assert config.n_actuators == 6
