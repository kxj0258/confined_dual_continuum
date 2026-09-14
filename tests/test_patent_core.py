"""专利核心不变量与少量跨层合同；不启动 MuJoCo。"""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET
import tempfile
import json
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from confined_dual_continuum.actuation.bending_space import BendingSpaceModel
from confined_dual_continuum.actuation.compatible_tendon_control import CompatibleTendonController, TendonLimits
from confined_dual_continuum.application.controller import ConfinedDualContinuumController, ControlInput
from confined_dual_continuum.coordination.interarm_safety import InterArmSafety
from confined_dual_continuum.coordination.observer_controller import ObserverController
from confined_dual_continuum.domain.geometry import Pose
from confined_dual_continuum.domain.state import ArmState, ArmCommand, RobotState
from confined_dual_continuum.kinematics.pcc import position_jacobian
from confined_dual_continuum.robot.configuration import load_robot_model
from confined_dual_continuum.task.confined_operation import OperationSignals
from confined_dual_continuum.task.operation_director import OperationDirector, OperationPlan
from confined_dual_continuum.task.phase_machine import ConfinedOperationPhase as Phase
from confined_dual_continuum.workspace.configuration import load_workspace
from confined_dual_continuum.workspace.query import Clearance


class EmptyEnvironment:
    def nearest_clearance(self, arm, *, allow_tool_contact):
        return Clearance(0.05, np.zeros(3), np.zeros(3), "empty")


def robot_state():
    def arm(y):
        return ArmState(Pose(np.array([0.,y,.14])), np.zeros(6), np.zeros(9),
                        np.array([[0.,y,.02], [0.,y,.14]]), np.zeros(3))
    return RobotState(Pose(np.zeros(3)), arm(.01), arm(-.01))


class PatentCoreTests(unittest.TestCase):
    def setUp(self):
        self.robot = load_robot_model(ROOT / "configs/robot/three_segment_nine_tendon.yaml")

    def controller(self):
        return ConfinedDualContinuumController(
            CompatibleTendonController(self.robot.bending_model("executor"), self.robot.limits),
            CompatibleTendonController(self.robot.bending_model("observer"), self.robot.limits),
            self.robot, EmptyEnvironment())

    def request(self, state, signals=OperationSignals()):
        return ControlInput(state, signals, state.executor.tip_pose.position + [.01,0,0],
                            state.observer.tip_pose.position + [.01,0,0], np.array([0.,0.,-1.]),
                            0., "observer", np.array([.1,0,0,0,0,0]), .01,
                            np.array([0.,0.,1.]), 0)

    def test_routes_match_xml_base_holes_for_both_arms(self):
        xml = ET.parse(ROOT / "assets/mujoco/multi_task_patent_demo.xml").getroot()
        for arm, routes in self.robot.tendon_routes.items():
            for i, route in enumerate(routes, start=1):
                tendon = xml.find(f"tendon/spatial[@name='{arm}_tendon_{i}']")
                site = xml.find(f".//site[@name='{tendon[0].attrib['site']}']")
                x,y,_ = map(float, site.attrib["pos"].split())
                angle = np.degrees(np.arctan2(y,x)) % 360
                self.assertAlmostEqual(angle, route.angle_deg, places=4)

    def test_shared_mount_gap_does_not_trigger_avoidance(self):
        state = robot_state()
        decision = InterArmSafety().evaluate(state.executor.centerline_world, state.observer.centerline_world)
        self.assertAlmostEqual(decision.distance_m, .0075)
        self.assertFalse(decision.avoidance_active)

    def test_crossing_segments_detected_between_samples(self):
        decision = InterArmSafety().evaluate(np.array([[-1.,0.,0.],[1.,0.,0.]]),
                                             np.array([[0.,-1.,0.],[0.,1.,0.]]))
        self.assertTrue(decision.safety_stop)
        self.assertLess(decision.distance_m, 0)

    def test_phase_transition_is_hold_then_new_phase_command(self):
        state = robot_state()
        controller = self.controller()
        controller.reset(state)
        first = controller.step(self.request(state, OperationSignals(True, "initialized")))
        np.testing.assert_array_equal(first.command.base_twist_world, np.zeros(6))
        np.testing.assert_array_equal(first.command.observer.tendon_rate_mps, np.zeros(9))
        second = controller.step(self.request(state))
        self.assertGreater(second.command.base_twist_world[0], 0)
        np.testing.assert_array_equal(second.command.observer.tendon_rate_mps, np.zeros(9))

    def test_disabled_arm_stays_disabled_during_avoidance(self):
        state = robot_state()
        observer = ArmState(Pose(np.array([0.,-.007,.14])), np.zeros(6), np.zeros(9),
                            np.array([[0.,-.007,.02],[0.,-.007,.14]]), np.zeros(3))
        state = RobotState(state.base_pose, state.executor, observer)
        controller = self.controller()
        controller.reset(state)
        controller.supervisor.machine.advance("initialized")
        output = controller.step(self.request(state))
        self.assertTrue(output.diagnostics.inter_arm_avoidance_active)
        np.testing.assert_array_equal(output.command.observer.tendon_rate_mps, np.zeros(9))

    def test_safety_stop_suppresses_all_commands(self):
        state = robot_state()
        state = RobotState(state.base_pose, state.executor, state.executor)
        controller = self.controller()
        controller.reset(state)
        output = controller.step(self.request(state))
        self.assertEqual(controller.phase, Phase.SAFETY_STOP)
        np.testing.assert_array_equal(output.command.base_twist_world, np.zeros(6))
        np.testing.assert_array_equal(output.command.executor.tendon_rate_mps, np.zeros(9))

    def test_tendon_limits_preserve_column_space(self):
        model = BendingSpaceModel(np.array([[1.,0.],[0.,1.],[-1.,-1.]]))
        limits = TendonLimits(np.full(3,-.1), np.full(3,.1), np.full(3,.2), np.full(3,.05))
        actuator = CompatibleTendonController(model, limits)
        step = actuator.step(np.array([2.,1.]), np.zeros(3), .1)
        self.assertTrue(step.feasible)
        self.assertTrue(model.is_compatible(step.tendon_rate_mps))
        self.assertTrue(np.all(np.abs(step.tendon_rate_mps) <= .2+1e-12))
        self.assertTrue(np.all(np.abs(step.target_m) <= .05+1e-12))

    def test_infeasible_lead_is_not_reported_as_success(self):
        from dataclasses import replace
        model = self.robot.bending_model("executor")
        actuator = CompatibleTendonController(model, replace(self.robot.limits, enabled=True))
        actuator.reset(np.zeros(9))
        step = actuator.step(np.zeros(6), np.full(9,.01), .01)
        self.assertFalse(step.feasible)
        np.testing.assert_array_equal(step.tendon_rate_mps, np.zeros(9))

    def test_zero_curvature_jacobian_has_finite_correct_limit(self):
        jacobian = position_jacobian(np.zeros(2), (.04,), (0.,))
        np.testing.assert_allclose(jacobian, np.array([[.0008,0],[0,.0008],[0,0]]), atol=1e-9)

    def test_observer_only_returns_its_six_bending_rates(self):
        rate = ObserverController().solve_view(np.zeros(3), np.ones(3)*.001,
            np.eye(3,6), np.array([0.,0.,1.]), np.array([0.,0.,1.]), np.zeros((3,6)))
        self.assertEqual(rate.shape, (6,))

    def test_unreached_path_times_out_instead_of_completing(self):
        workspace = load_workspace(ROOT / "configs/workspace/multi_task_patent_bench.yaml")
        plan = OperationPlan(.005, 1.5, .1, .02, 1., .025, np.array([0.,.01,.14]), waypoint_timeout_s=.1)
        director = OperationDirector(workspace, plan)
        for i in range(15):
            output = director.setpoint(Phase.CONFINED_VISUAL_INSPECTION, i, robot_state(), .01)
        self.assertFalse(output.signals.phase_complete)
        self.assertEqual(output.signals.safety_violation, "waypoint_or_transit_timeout")

    def test_nonfinite_commands_rejected(self):
        with self.assertRaises(ValueError):
            ArmCommand(np.array([np.nan]), np.zeros(1))

    def test_cancelled_step_restores_integrator(self):
        actuator = CompatibleTendonController(self.robot.bending_model("executor"), self.robot.limits)
        actuator.reset(np.zeros(9))
        result = actuator.step(np.ones(6), np.zeros(9), .01)
        actuator.cancel_step(result, .01)
        next_result = actuator.step(np.zeros(6), np.zeros(9), .01)
        np.testing.assert_allclose(next_result.target_m, np.zeros(9), atol=1e-15)

    def test_final_execution_failure_overrides_complete(self):
        controller = self.controller()
        controller.supervisor.machine.phase = Phase.COMPLETE
        controller.supervisor.update(OperationSignals(safety_violation="infeasible"))
        self.assertEqual(controller.phase, Phase.SAFETY_STOP)

    def test_backend_failure_saves_summary_and_stops(self):
        from confined_dual_continuum.application.session import OperationSession, SessionConfig
        class FailingBackend:
            stopped = False
            def reset(self): return robot_state()
            def apply_command(self, command): pass
            def step(self, dt): raise RuntimeError("backend_failure")
            def stop(self): self.stopped = True
        backend = FailingBackend()
        workspace = load_workspace(ROOT / "configs/workspace/multi_task_patent_bench.yaml")
        plan = OperationPlan(.005, 1.5, .1, .02, 1., .025, np.array([0.,.01,.14]))
        session = OperationSession(backend, self.controller(), OperationDirector(workspace, plan), SessionConfig(.01,.02))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "backend_failure"):
                session.run(directory)
            summary = json.loads((Path(directory)/"summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["completed"])
            self.assertIn("backend_failure", summary["termination_reason"])
            self.assertTrue(backend.stopped)

    def test_formal_assembly_uses_compatible_actuation(self):
        import yaml
        from confined_dual_continuum import cli
        from confined_dual_continuum.actuation.independent_tendon_control import IndependentTendonController
        class Backend(EmptyEnvironment):
            def stop(self): pass
            def close(self): pass
        path = ROOT / "configs/experiments/unified_demo.yaml"
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        with patch.object(cli, "_snapshot"), patch.object(cli, "MujocoRobotBackend", return_value=Backend()), patch.object(cli, "_run", side_effect=lambda session, *args: session):
            formal = cli.run_experiment(path, values, ROOT, False, False)
            self.assertIs(type(formal.controller.executor_actuation), CompatibleTendonController)
            self.assertTrue(formal.controller.staged_activation)
            baseline = cli.run_experiment(path, values, ROOT, False, False, "independent_clipping")
            self.assertIs(type(baseline.controller.executor_actuation), IndependentTendonController)


if __name__ == "__main__":
    unittest.main()
