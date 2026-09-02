"""专利核心只保留跨层不变量测试，避免绑定内部实现细节。"""

import numpy as np

from confined_dual_continuum.actuation.bending_space import BendingSpaceModel
from confined_dual_continuum.actuation.compatible_tendon_control import (
    CompatibleTendonController,
    TendonLimits,
)
from confined_dual_continuum.coordination.observer_target import ObserverTargetGenerator
from confined_dual_continuum.kinematics.pcc import forward_kinematics
from confined_dual_continuum.task.confined_operation import ConfinedOperationSupervisor, OperationSignals


def test_phase_activation_follows_patent_sequence() -> None:
    supervisor = ConfinedOperationSupervisor()
    decision = supervisor.update(OperationSignals(initialized=True))
    assert decision.phase.value == "base_approach"
    assert decision.activation.base_enabled
    assert not decision.activation.executor_enabled


def test_observer_target_uses_surface_geometry() -> None:
    target = ObserverTargetGenerator().generate(
        np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 0.02
    )
    np.testing.assert_allclose(target.position_world, [-0.08, 0.02, 0.0])


def test_tendon_output_remains_compatible_after_limiting() -> None:
    coupling = np.array([[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]])
    model = BendingSpaceModel(coupling)
    limits = TendonLimits(
        np.full(3, -0.01), np.full(3, 0.01), np.full(3, 0.001), np.full(3, 0.001)
    )
    result = CompatibleTendonController(model, limits).step(
        np.array([10.0, -5.0]), np.zeros(3), 0.02
    )
    assert model.is_compatible(result.tendon_rate_mps)
    assert np.max(np.abs(result.tendon_rate_mps)) <= 0.001


def test_straight_pcc_arm_has_expected_length() -> None:
    result = forward_kinematics(np.zeros(6), (0.0365, 0.0365, 0.0365))
    np.testing.assert_allclose(result.tip_transform[:3, 3], [0.0, 0.0, 0.1095])
