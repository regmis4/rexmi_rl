"""Shared policy loading and velocity injection used by teleop and navigation."""
import os
from typing import Callable

POLICY_DEFAULTS = {
    "rough": {
        "vx": 0.45,
        "omega": 0.40,
        "ang_range": (-1.0, 1.0),
        "lin_x_range": (-0.5, 1.0),
    },
    "rocky_slope": {
        "vx": 0.40,
        "omega": 0.35,
        "ang_range": (-1.0, 1.0),
        "lin_x_range": (-0.5, 1.0),
    },
    "turn": {
        # model_13345 Language A band (isolation / reorient)
        "vx": 0.05,
        "omega": 0.07,
        "ang_range": (-0.10, 0.10),
        "lin_x_range": (0.0, 0.15),
    },
}

def load_policy(checkpoint_path: str, device: str) -> Callable:
    import torch
    from rsl_rl.modules import ActorCritic

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    msd = ckpt["model_state_dict"]
    obs_dim = msd["actor.0.weight"].shape[1]
    num_actions = msd["actor.6.weight"].shape[0]
    hidden_dims = [
        msd["actor.0.bias"].shape[0],
        msd["actor.2.bias"].shape[0],
        msd["actor.4.bias"].shape[0],
    ]
    ac = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=num_actions,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
    ).to(device)
    ac.load_state_dict(msd)
    ac.eval()

    obs_mean = obs_std = None
    if "obs_norm_state_dict" in ckpt:
        nd = ckpt["obs_norm_state_dict"]
        if "_mean" in nd and "_std" in nd:
            obs_mean = nd["_mean"].to(device)
            obs_std = nd["_std"].to(device).clamp(min=1e-6)
            print(f"[teleop] obs normalizer loaded shape={tuple(obs_mean.shape)}")

    def policy(obs):
        with torch.inference_mode():
            o = obs[..., :obs_dim]
            if obs_mean is not None:
                o = (o - obs_mean) / obs_std
            return ac.act_inference(o)

    print(
        f"[teleop] Loaded {os.path.basename(checkpoint_path)} "
        f"obs={obs_dim} act={num_actions} hidden={hidden_dims}"
    )
    return policy


def inject_command(env, vx: float, omega: float, env_idx: int = 0) -> None:
    try:
        cmd = env.unwrapped.command_manager.get_command("base_velocity")
        cmd[env_idx, 0] = float(vx)
        cmd[env_idx, 1] = 0.0
        cmd[env_idx, 2] = float(omega)
    except Exception as e:
        raise RuntimeError("Velocity command injection failed") from e


def patch_command_ranges(env, policy_name: str) -> None:
    d = POLICY_DEFAULTS[policy_name]
    try:
        bv = env.unwrapped.command_manager.cfg.base_velocity
        bv.ranges.ang_vel_z = d["ang_range"]
        if hasattr(bv.ranges, "lin_vel_x"):
            bv.ranges.lin_vel_x = d["lin_x_range"]
        # Disable heading P-controller so injected ω is used as-is
        if hasattr(bv, "heading_command"):
            bv.heading_command = False
        if hasattr(bv, "rel_heading_envs"):
            bv.rel_heading_envs = 0.0
        print(
            f"[teleop] command ranges for {policy_name}: "
            f"ang={d['ang_range']} lin_x={d['lin_x_range']}"
        )
    except Exception as e:
        print(f"[teleop] WARNING range patch: {e}")




def directional_command(forward=False, back=False, left=False, right=False, vx=0.4, omega=0.35):
    """Body forward/back; positive yaw is left. Stop keeps the policy active."""
    return (vx if forward else -vx if back else 0.0,
            omega if left else -omega if right else 0.0)


def policy_actions(env, policy, vx, omega, env_idx=0):
    """Inject before observation/inference, without advancing observation history.

    Both teleop and checkpoint control use this ordering so Stop reaches the
    policy on this tick, rather than using the preceding tick's command.
    """
    inject_command(env, vx, omega, env_idx)
    obs = env.unwrapped.observation_manager.compute(update_history=False)['policy']
    return policy(obs)
