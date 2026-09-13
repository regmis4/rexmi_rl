"""CPU-only logger tests: python -m unittest discover -s tests -v."""

import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from joint_telemetry import JointTelemetry
from teleop import TeleopState


class Config:
    decimation = 4

    def to_dict(self):
        return {"decimation": self.decimation}


class Scene:
    num_envs = 1

    def __init__(self, robot, sensor):
        self.robot = robot
        self.sensors = {"contact_forces": sensor} if sensor else {}
        self.updates = 0

    def __getitem__(self, name):
        return self.robot

    def update(self, dt):
        self.updates += 1


def fake_env(contacts=True):
    data = NS(**{attr: [[1.0, 2.0]] for attr in JointTelemetry.FIELDS.values()})
    data.computed_torque = [[10.0, -4.0]]
    data.applied_torque = [[8.0, -4.0]]
    data.joint_vel = [[2.0, 3.0]]
    data.root_pos_w = [[1.0, 2.0, 3.0]]
    data.root_quat_w = [[1.0, 0.0, 0.0, 0.0]]
    robot = NS(data=data, joint_names=["FL_calf_joint", "RR_wheel_joint"], actuators={})
    sensor = NS(body_names=["FL_wheel", "RR_wheel", "base"],
                data=NS(net_forces_w=[[[0.0, 0.0, 20.0], [0.0, 0.0, 2.0], [0.0, 0.0, 50.0]]]))
    return NS(scene=Scene(robot, sensor if contacts else None), cfg=Config(),
              physics_dt=0.005, step_dt=0.02)


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "run"
        self.env = fake_env()

    def logger(self):
        logger = JointTelemetry(self.path, self.env, metadata={})
        self.addCleanup(logger.close)
        return logger

    def step(self, logger, step=0, label="uphill", done=False):
        logger.begin_step(step, "rough", label, 0.4, 0.0)
        for _ in range(4):
            self.env.scene.update(dt=0.005)
        logger.end_step(done, terminated=done, truncated=False)

    def rows(self):
        with (self.path / "samples.csv").open() as stream:
            return list(csv.DictReader(stream))

    def test_physics_rate_power_contacts_and_restore(self):
        original = self.env.scene.update
        logger = self.logger()
        self.env.scene.update(dt=0.005)  # Non-control update is not captured.
        self.step(logger)
        logger.close()
        rows = self.rows()
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(float(rows[-1]["sim_time_s"]), 0.02)
        self.assertEqual(rows[-1]["substep"], "3")
        self.assertEqual(rows[0]["FL_calf_joint.power_est_W"], "16.0")
        self.assertEqual(rows[0]["RR_wheel_joint.power_est_W"], "-12.0")
        self.assertEqual(rows[0]["FL_calf_joint.torque_clipped_est"], "1")
        self.assertEqual(rows[0]["RR_wheel_joint.torque_clipped_est"], "0")
        self.assertEqual(rows[0]["wheel_contact_count_est"], "1")
        self.assertEqual(self.env.scene.update, original)
        self.assertEqual(self.env.scene.updates, 5)

    def test_reset_boundaries_and_scenario_labels(self):
        logger = self.logger()
        self.step(logger, done=True)
        # Mimic auto-reset changing live buffers AFTER the terminal samples.
        self.env.scene.robot.data.joint_pos = [[99.0, 99.0]]
        self.step(logger, step=1, label="downhill, slow")
        logger.close()
        rows = self.rows()
        self.assertEqual(rows[3]["episode"], "0")
        self.assertEqual(rows[3]["FL_calf_joint.position_rad"], "1.0")
        self.assertEqual(rows[4]["episode"], "1")
        self.assertEqual(rows[4]["scenario"], "downhill, slow")
        events = [json.loads(line) for line in (self.path / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(e["event"] == "auto_reset" for e in events), 1)
        self.assertTrue(next(e for e in events if e["event"] == "auto_reset")["terminated"])

    def test_optional_data_missing_is_blank(self):
        self.env = fake_env(contacts=False)
        self.env.scene.robot.data.joint_acc = None
        logger = self.logger()
        self.step(logger)
        logger.close()
        row = self.rows()[0]
        self.assertEqual(row["FL_calf_joint.acceleration_rad_s2"], "")
        self.assertEqual(row["wheel_contact_count_est"], "")
        info = json.loads((self.path / "metadata.json").read_text())
        self.assertEqual(info["joint_names"], self.env.scene.robot.joint_names)
        self.assertIn("NOT measured", info["torque_semantics"])

    def test_no_overwrite(self):
        logger = self.logger()
        with self.assertRaises(FileExistsError):
            JointTelemetry(self.path, self.env, metadata={})
        logger.close()

    def test_missing_substeps_fails_loudly(self):
        logger = self.logger()
        logger.begin_step(0, "rough", "test", 0, 0)
        self.env.scene.update(dt=0.005)
        with self.assertRaisesRegex(RuntimeError, "expected 4"):
            logger.end_step()

    def test_missing_required_torque_fails_loudly(self):
        logger = self.logger()
        self.env.scene.robot.data.applied_torque = None
        logger.begin_step(0, "rough", "test", 0, 0)
        with self.assertRaisesRegex(RuntimeError, "applied_torque"):
            self.env.scene.update(dt=0.005)

    def test_scenario_does_not_change_control(self):
        state = TeleopState()
        state.set_hold("forward", True)
        before = state.command()
        state.set_scenario("  slope 35  ")
        self.assertEqual(state.scenario_label(), "slope 35")
        self.assertEqual(state.command(), before)


if __name__ == "__main__":
    unittest.main()
