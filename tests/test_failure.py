"""
Tests for single-actuator failure analysis.

These cover the DOF-loss detection that distinguishes this tool from a plain
rank check: the point is to report *which* axis a failure costs.
"""

from __future__ import annotations

import numpy as np

from actuator_viz import (
    Actuator,
    ActuatorConfig,
    achievable_dofs,
    analyze_all_failures,
    compute_effectiveness_matrix,
    simulate_failure,
)


def test_achievable_dofs_empty_matrix():
    assert achievable_dofs(np.zeros((6, 0))) == set()


def test_achievable_dofs_single_force(single_z_thruster):
    e = compute_effectiveness_matrix(single_z_thruster)
    assert achievable_dofs(e) == {"Fz"}


def test_achievable_dofs_three_forces(three_axis_forces):
    e = compute_effectiveness_matrix(three_axis_forces)
    assert achievable_dofs(e) == {"Fx", "Fy", "Fz"}


def test_losing_unique_contributor_is_critical(forces_with_redundant_z):
    # Actuator 0 is the only source of Fx -> its loss removes Fx.
    impact = simulate_failure(forces_with_redundant_z, actuator_index=0)
    assert impact.actuator_id == 0
    assert impact.lost_dofs == ["Fx"]
    assert impact.critical


def test_losing_redundant_actuator_is_non_critical(forces_with_redundant_z):
    # Actuators 2 and 3 both supply Fz, so losing one keeps full capability.
    impact = simulate_failure(forces_with_redundant_z, actuator_index=2)
    assert impact.lost_dofs == []
    assert not impact.critical


def test_analyze_all_failures_one_per_actuator(forces_with_redundant_z):
    impacts = analyze_all_failures(forces_with_redundant_z)
    assert [i.actuator_id for i in impacts] == [0, 1, 2, 3]
    # Exactly the two unique force axes (Fx, Fy) are single points of failure.
    critical_ids = {i.actuator_id for i in impacts if i.critical}
    assert critical_ids == {0, 1}


def test_over_actuated_rov_tolerates_any_single_failure(rov_config):
    impacts = analyze_all_failures(rov_config)
    assert len(impacts) == rov_config.n_actuators
    assert all(not i.critical for i in impacts)
    assert all(i.controllable for i in impacts)


def test_single_actuator_failure_loses_everything():
    # Removing the only actuator leaves nothing controllable.
    config = ActuatorConfig(
        actuators=[
            Actuator(id=7, position=(0, 0, 0), axis=(0, 0, 1)),
        ]
    )
    impact = simulate_failure(config, actuator_index=0)
    assert impact.rank == 0
    assert not impact.controllable
    assert impact.condition_number == float("inf")
    assert impact.lost_dofs == ["Fz"]
