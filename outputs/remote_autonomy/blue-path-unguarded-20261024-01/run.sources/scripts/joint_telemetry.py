"""Optional, single-environment teleop telemetry. No Isaac imports or control writes.

The scene-update hook runs after physics/buffer updates, before RL auto-reset.
Implicit actuator efforts are PD estimates evaluated at the start of a substep;
positions/velocities are sampled at its end. Power is therefore an estimate too.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def plain(value):
    """Serialize config/tensor values without importing torch or Isaac."""
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def vector(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def fingerprint(path):
    path = Path(path).expanduser().resolve()
    result = {"path": str(path), "exists": path.is_file()}
    if result["exists"]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


class JointTelemetry:
    """Stream one wide CSV row per physics step; bounded memory, no dropped rows.

    Attach only after settling. Call begin_step before env.step and end_step
    afterwards. Missing physics samples are a hard error, not silent downsampling.
    """

    FIELDS = {
        "position_rad": "joint_pos",
        "velocity_rad_s": "joint_vel",
        "acceleration_rad_s2": "joint_acc",
        "position_target_rad": "joint_pos_target",
        "velocity_target_rad_s": "joint_vel_target",
        "torque_demand_est_Nm": "computed_torque",
        "torque_applied_est_Nm": "applied_torque",
        "effort_limit_Nm": "joint_effort_limits",
        "velocity_limit_rad_s": "joint_vel_limits",
    }
    ROOT_FIELDS = {
        "root_position_w_m": ("root_pos_w", ("x", "y", "z")),
        "root_quaternion_w": ("root_quat_w", ("w", "x", "y", "z")),
        "root_linear_velocity_w_m_s": ("root_lin_vel_w", ("x", "y", "z")),
        "root_angular_velocity_w_rad_s": ("root_ang_vel_w", ("x", "y", "z")),
    }

    def __init__(self, directory, env, *, metadata, contact_threshold_N=5.0):
        self.directory = Path(directory).expanduser().resolve()
        # Never overwrite an earlier characterization run.
        self.directory.mkdir(parents=True, exist_ok=False)
        self.env = env
        self.robot = env.scene["robot"]
        if env.scene.num_envs != 1:
            raise ValueError("Teleop telemetry requires exactly one environment")
        self.names = list(self.robot.joint_names)
        self.threshold = contact_threshold_N
        self.sensor = env.scene.sensors.get("contact_forces")
        self.contact_names = list(self.sensor.body_names) if self.sensor is not None else []
        self.wheel_indices = [i for i, name in enumerate(self.contact_names) if "wheel" in name.lower()]
        self.sample = 0
        self.elapsed = 0.0
        self.episode = 0
        self.active = False
        self.closed = False
        self.context = {}
        self.last_flush = time.monotonic()
        self._original_update = env.scene.update
        self._had_instance_update = "update" in vars(env.scene)
        self._instance_update = vars(env.scene).get("update")

        header = ["sample", "sim_time_s", "dt_s", "episode", "control_step", "substep",
                  "policy", "scenario", "vx_command_m_s", "omega_command_rad_s"]
        for prefix, (_, axes) in self.ROOT_FIELDS.items():
            header.extend(f"{prefix}.{axis}" for axis in axes)
        header.append("wheel_contact_count_est")
        for name in self.names:
            header.extend(f"{name}.{field}" for field in self.FIELDS)
            header.extend(f"{name}.{field}" for field in ("power_est_W", "torque_clipped_est"))
        for name in self.contact_names:
            header.extend(f"contact.{name}.{axis}_N" for axis in ("fx_w", "fy_w", "fz_w"))

        info = dict(metadata)
        info.update({
            "schema_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "joint_names": self.names, "contact_body_names": self.contact_names,
            "wheel_contact_body_indices": self.wheel_indices,
            "contact_threshold_N": self.threshold, "physics_dt_s": env.physics_dt,
            "control_dt_s": env.step_dt, "decimation": env.cfg.decimation,
            "resolved_env_config": env.cfg.to_dict(),
            "torque_semantics": "Actuator-model telemetry, NOT measured motor torque. Implicit PD estimates at substep start; state at substep end. Power = estimated applied torque * end velocity.",
            "sampling": "Every scene.update during env.step, after physics, before auto-reset. Startup/settling excluded. Time relative to capture start.",
            "contact_semantics": "World-frame net body contact force, not motor/bearing reaction. Wheel contact = norm > threshold; may include side/obstacle contacts, not guaranteed ground support.",
            "missing_values": "Empty CSV field means unavailable; never interpret as zero.",
            "columns": header,
        })
        info["actuators"] = {
            name: {"class": type(actuator).__name__,
                   "joint_names": actuator.joint_names,
                   "config": actuator.cfg.to_dict()}
            for name, actuator in self.robot.actuators.items()
        }
        info["runtime_properties"] = {}
        for attr in ("joint_stiffness", "joint_damping", "joint_armature", "joint_pos_limits",
                     "default_mass", "default_inertia"):
            value = getattr(self.robot.data, attr, None)
            info["runtime_properties"][attr] = None if value is None else vector(value[0])
        try:
            info["live_body_masses_kg"] = vector(self.robot.root_physx_view.get_masses()[0])
            info["body_names"] = list(self.robot.body_names)
        except AttributeError:
            info["live_body_masses_kg"] = None
        repo = Path(__file__).resolve().parents[1]
        for key, command in (("git_commit", ["git", "rev-parse", "HEAD"]),
                             ("git_status", ["git", "status", "--porcelain"])):
            result = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
            info[key] = result.stdout.strip() if result.returncode == 0 else None
        for filename in ("teleop.py", "joint_telemetry.py"):
            info.setdefault("logger_sources", {})[filename] = fingerprint(Path(__file__).with_name(filename))
        with (self.directory / "metadata.json").open("x") as stream:
            json.dump(info, stream, indent=2, default=plain)
        self.stream = (self.directory / "samples.csv").open("x", newline="")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(header)
        self.events = (self.directory / "events.jsonl").open("x", buffering=1)
        env.scene.update = self._after_update
        self.event("capture_start")

    def event(self, kind, **details):
        self.events.write(json.dumps({"event": kind, "sim_time_s": self.elapsed,
                                     "episode": self.episode, **details}) + "\n")

    def begin_step(self, step, policy, scenario, vx, omega):
        if self.active:
            raise RuntimeError("Previous telemetry step not finished")
        context = dict(control_step=step, policy=policy, scenario=scenario,
                       vx_command_m_s=vx, omega_command_rad_s=omega)
        if any(context[k] != self.context.get(k) for k in ("policy", "scenario")):
            self.event("segment", **context)
        self.context = context
        self.substep = 0
        self.active = True

    def _after_update(self, *args, **kwargs):
        result = self._original_update(*args, **kwargs)
        if self.active:
            dt = kwargs.get("dt", args[0] if args else self.env.physics_dt)
            self._capture(float(dt))
        return result

    def _capture(self, dt):
        data = self.robot.data
        values = {}
        for label, attr in self.FIELDS.items():
            value = getattr(data, attr, None)
            if value is None and attr in ("joint_pos", "joint_vel", "computed_torque", "applied_torque"):
                raise RuntimeError(f"Required telemetry unavailable: {attr}")
            values[label] = [None] * len(self.names) if value is None else vector(value[0])
        self.elapsed += dt
        row = [self.sample, self.elapsed, dt, self.episode, self.context["control_step"],
               self.substep, self.context["policy"], self.context["scenario"],
               self.context["vx_command_m_s"], self.context["omega_command_rad_s"]]
        for attr, axes in self.ROOT_FIELDS.values():
            value = getattr(data, attr, None)
            row.extend([None] * len(axes) if value is None else vector(value[0]))
        forces = vector(self.sensor.data.net_forces_w[0]) if self.sensor is not None else []
        row.append(sum(math.sqrt(sum(v * v for v in forces[i])) > self.threshold
                       for i in self.wheel_indices) if self.wheel_indices else None)
        for i in range(len(self.names)):
            row.extend(values[field][i] for field in self.FIELDS)
            applied = values["torque_applied_est_Nm"][i]
            demand = values["torque_demand_est_Nm"][i]
            speed = values["velocity_rad_s"][i]
            row.extend((applied * speed, int(abs(demand - applied) > 1e-5)))
        for force in forces:
            row.extend(force)
        self.writer.writerow(row)
        self.sample += 1
        self.substep += 1
        if time.monotonic() - self.last_flush >= 1.0:
            self.stream.flush()
            self.last_flush = time.monotonic()

    def end_step(self, done=False, terminated=None, truncated=None):
        self.active = False
        if self.substep != self.env.cfg.decimation:
            raise RuntimeError(f"Telemetry saw {self.substep} physics steps; expected {self.env.cfg.decimation}")
        if done:
            self.event("auto_reset", control_step=self.context["control_step"],
                       terminated=terminated, truncated=truncated)
            self.episode += 1

    def close(self):
        if self.closed:
            return
        self.active = False
        if self._had_instance_update:
            self.env.scene.update = self._instance_update
        else:
            del self.env.scene.update
        self.event("capture_end", samples=self.sample)
        self.stream.close()
        self.events.close()
        self.closed = True
