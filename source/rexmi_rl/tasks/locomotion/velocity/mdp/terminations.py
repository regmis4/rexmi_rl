# Copyright (c) 2026, REXMI Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination terms for REXMI velocity-tracking environments.

WHY THESE EXIST
---------------
Isaac Lab's stock `bad_orientation` collapses pitch and roll into one angle
between body-z and world-up. On a slope-turn policy that sometimes somersaults
(pitch) and sometimes falls sideways (roll), a single number cannot tell us
which failure mode is dominant — so we cannot tell whether the next reward
change should target front/rear load transfer or lateral CoM shift.

These terms split that signal. They use projected_gravity_b the same way
`trunk_stability_penalty` does:

    grav_b[0] ~ sin(pitch)   (nose-down positive)
    grav_b[1] ~ sin(roll)    (roll-right positive)

`bad_pitch` / `bad_roll` are real terminations (episode ends).
`log_abs_pitch` / `log_abs_roll` are non-terminating helpers intended for
zero-weight reward terms that surface under Episode_Reward/* as diagnostics.
NOTE: zero-weight rewards log zeros in Isaac Lab's RewardManager because it
accumulates func()*weight*dt. Prefer reading the termination rates for the
real signal; the log_* helpers are only useful if wired as small non-zero
weights or via a custom metric path.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def bad_pitch(
    env: ManagerBasedRLEnv,
    limit_angle: float = 1.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when |pitch| exceeds limit_angle (radians).

    Pitch is recovered as asin(projected_gravity_b_x), i.e. the body-frame
    forward component of gravity. Somersault / nose-dive failures trip this;
    pure sideways rolls do not.
    """
    asset = env.scene[asset_cfg.name]
    # clamp for numerical safety near +/-1
    s = asset.data.projected_gravity_b[:, 0].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    pitch = torch.asin(s).abs()
    return pitch > limit_angle


def bad_roll(
    env: ManagerBasedRLEnv,
    limit_angle: float = 1.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when |roll| exceeds limit_angle (radians).

    Roll is recovered as asin(projected_gravity_b_y). Sideways topples trip
    this; pure pitch somersaults do not.
    """
    asset = env.scene[asset_cfg.name]
    s = asset.data.projected_gravity_b[:, 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    roll = torch.asin(s).abs()
    return roll > limit_angle


def log_abs_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """|pitch| in radians. Diagnostic helper, not a termination."""
    asset = env.scene[asset_cfg.name]
    s = asset.data.projected_gravity_b[:, 0].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.asin(s).abs()


def log_abs_roll(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """|roll| in radians. Diagnostic helper, not a termination."""
    asset = env.scene[asset_cfg.name]
    s = asset.data.projected_gravity_b[:, 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.asin(s).abs()


def log_horizontal_drift(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Horizontal metres from this episode's spawn XY.

    Reuses the same spawn-tracking pattern as position_drift_penalty so the
    number is comparable. Diagnostic helper.
    """
    robot = env.scene["robot"]
    pos_w = robot.data.root_pos_w[:, :2]
    is_first = env.episode_length_buf == 1
    if "drift_spawn_xy" not in env.extras:
        env.extras["drift_spawn_xy"] = pos_w.clone()
    spawn = env.extras["drift_spawn_xy"]
    spawn = torch.where(is_first.unsqueeze(1).expand_as(spawn), pos_w, spawn)
    env.extras["drift_spawn_xy"] = spawn
    return (pos_w - spawn).norm(dim=-1)
