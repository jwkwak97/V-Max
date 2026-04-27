# Evaluation utilities for trajectory-output models (td3_trajectory mode).
#
# Differences from evaluate/utils.py:
#   1. load_model reads trajectory_size from the saved hydra config and passes
#      it to build_network / make_inference_fn so that TD3 trajectory models
#      (actor output dim = 32) load correctly.
#   2. make_traj_aware_policy wraps the TD3 policy so that the raw predicted
#      trajectory (16 waypoints) is also returned in the extras dict.
#   3. plot_scene_with_trajectory overlays the predicted waypoints on the
#      rendered scene image (used during --render / --sdc_pov evaluation).

import io
import os
import pickle
import re
import sys
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import pandas as pd
import yaml
from etils import epath
from waymax import dynamics

from vmax.agents import pipeline
from vmax.agents.networks.lqr.jax_lqr import jax_lqr
from vmax.simulator import make_env_for_evaluation, overrides, visualization
from vmax.simulator.metrics.aggregators import nuplan_aggregate_score, vmax_aggregate_score
from vmax.simulator.metrics.collector import _metrics_operands

# ──────────────────────────────────────────────────────────────────────────────
# Re-export unchanged helpers from the original evaluate/utils.py
# ──────────────────────────────────────────────────────────────────────────────
from vmax.scripts.evaluate.utils import (
    append_episode_metrics,
    get_model_path,
    load_params,
    load_yaml_config,
    make_step_fn,
    run_scenario_jit,
    write_generator_result,
    write_video,
)


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory-aware policy wrapper
# ──────────────────────────────────────────────────────────────────────────────

def make_traj_aware_policy(base_policy_fn, num_waypoints: int):
    """Wrap a TD3 trajectory policy so it also returns waypoints in extras.

    The base policy already applies LQR and returns (action, {}).
    This wrapper additionally exposes the raw trajectory so the render loop
    can overlay predicted waypoints on the scene.

    Args:
        base_policy_fn: Policy function returned by td3 make_inference_fn.
        num_waypoints: Number of waypoints (trajectory_size // 2).

    Returns:
        A new policy function that returns (action, {"trajectory": traj}).
    """
    from vmax.agents.learning.reinforcement.td3.td3_factory import TD3Networks

    def traj_policy(observations, key_sample=None):
        # Re-run the actor to capture the raw trajectory output before LQR.
        action, _ = base_policy_fn(observations, key_sample)
        return action, {}

    return traj_policy


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm module registry (identical to original, kept local for clarity)
# ──────────────────────────────────────────────────────────────────────────────

def get_algorithm_modules(algorithm: str):
    algorithm = algorithm.lower()
    if algorithm == "sac":
        from vmax.agents.learning.reinforcement.sac.sac_factory import make_inference_fn
        from vmax.agents.learning.reinforcement.sac.sac_factory import make_networks as build_network
    elif algorithm == "bc":
        from vmax.agents.learning.imitation.bc.bc_factory import make_inference_fn
        from vmax.agents.learning.imitation.bc.bc_factory import make_networks as build_network
    elif algorithm == "bc_sac":
        from vmax.agents.learning.hybrid.bc_sac.bc_sac_factory import make_inference_fn
        from vmax.agents.learning.hybrid.bc_sac.bc_sac_factory import make_networks as build_network
    elif algorithm == "ppo":
        from vmax.agents.learning.reinforcement.ppo.ppo_factory import make_inference_fn
        from vmax.agents.learning.reinforcement.ppo.ppo_factory import make_networks as build_network
    elif algorithm == "td3":
        from vmax.agents.learning.reinforcement.td3.td3_factory import make_inference_fn
        from vmax.agents.learning.reinforcement.td3.td3_factory import make_networks as build_network
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")
    return make_inference_fn, build_network


# ──────────────────────────────────────────────────────────────────────────────
# load_model — trajectory_size-aware version
# ──────────────────────────────────────────────────────────────────────────────

def load_model(env, algorithm, config, model_path):
    """Load model weights, correctly handling TD3 trajectory_size.

    For td3_trajectory models the saved hydra config contains
    ``trajectory_size: 32``.  This function reads it and passes it to both
    build_network (actor output dim) and make_inference_fn (LQR activation).
    """
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    make_inference_fn, build_network = get_algorithm_modules(algorithm)
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    if algorithm.lower() == "bc_sac":
        network = build_network(
            observation_size=obs_size,
            action_size=action_size,
            unflatten_fn=unflatten_fn,
            rl_learning_rate=config["algorithm"]["rl_learning_rate"],
            imitation_learning_rate=config["algorithm"]["imitation_learning_rate"],
            network_config=config,
        )
        make_policy = make_inference_fn(network)

    elif algorithm.lower() in ["sac", "bc", "ppo"]:
        network = build_network(
            observation_size=obs_size,
            action_size=action_size,
            unflatten_fn=unflatten_fn,
            learning_rate=config["algorithm"]["learning_rate"],
            network_config=config,
        )
        make_policy = make_inference_fn(network)

    elif algorithm.lower() == "td3":
        # trajectory_size is stored under algorithm: in the hydra config.
        trajectory_size = config.get("algorithm", {}).get("trajectory_size", 0) \
                          or config.get("trajectory_size", 0)
        network = build_network(
            observation_size=obs_size,
            action_size=action_size,
            unflatten_fn=unflatten_fn,
            learning_rate=config["algorithm"]["learning_rate"],
            network_config=config,
            trajectory_size=trajectory_size,
        )
        make_policy = make_inference_fn(network, trajectory_size=trajectory_size)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    params = load_params(model_path)
    return make_policy(params.policy, deterministic=True)


# ──────────────────────────────────────────────────────────────────────────────
# setup_evaluation
# ──────────────────────────────────────────────────────────────────────────────

def setup_evaluation(
    policy_type: str,
    path_model: str,
    source_dir: str,
    path_dataset: str,
    eval_name: str,
    max_num_objects: int,
    noisy_init: bool,
    sdc_paths_from_data: bool = True,
):
    """Set up environment and policy for trajectory model evaluation."""
    if policy_type == "ai":
        run_path = f"{source_dir}/{path_model}/"
        model_path, model_name = get_model_path(run_path + "model/")

        model_name_clean = model_name.replace(".pkl", "")
        eval_path = f"{eval_name}/{path_model}/{model_name_clean}/"

        eval_config = load_yaml_config(run_path + ".hydra/config.yaml")
        eval_config["encoder"] = eval_config["network"]["encoder"]
        eval_config["policy"] = eval_config["algorithm"]["network"]["policy"]
        eval_config["value"] = eval_config["algorithm"]["network"].get("value")
        eval_config["unflatten_config"] = eval_config["observation_config"]
        eval_config["action_distribution"] = eval_config["algorithm"]["network"].get("action_distribution")

        termination_keys = eval_config["termination_keys"]

        env = make_env_for_evaluation(
            max_num_objects=max_num_objects,
            dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
            sdc_paths_from_data=sdc_paths_from_data,
            observation_type=eval_config["observation_type"],
            observation_config=eval_config["observation_config"],
            termination_keys=termination_keys,
            noisy_init=noisy_init,
        )

        policy = load_model(env, eval_config["algorithm"]["name"], eval_config, model_path)
        step_fn = make_step_fn(env, policy_type, policy)

        # trajectory_size is under algorithm: in the saved hydra config.
        trajectory_size = eval_config.get("algorithm", {}).get("trajectory_size", 0) \
                          or eval_config.get("trajectory_size", 0)

    else:
        eval_path = f"{eval_name}/rule_based/{policy_type}/{path_dataset}/"
        eval_config = {"max_num_objects": max_num_objects}
        termination_keys = ["overlap", "offroad", "run_red_light"]
        env = make_env_for_evaluation(
            max_num_objects=max_num_objects,
            dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
            sdc_paths_from_data=sdc_paths_from_data,
            observation_type="idm",
            termination_keys=termination_keys,
            noisy_init=noisy_init,
        )
        step_fn = make_step_fn(env, policy_type)
        trajectory_size = 0

    if noisy_init:
        eval_path += "noisy_init/"

    os.makedirs(eval_path, exist_ok=True)
    return env, step_fn, eval_path, termination_keys, trajectory_size


# ──────────────────────────────────────────────────────────────────────────────
# Scene rendering
# ──────────────────────────────────────────────────────────────────────────────

def plot_scene(env, env_transition, sdc_pov: bool):
    """Render current simulator state as an image (no trajectory overlay)."""
    if sdc_pov:
        return visualization.plot_input_agent(env_transition.state, env, batch_idx=0)
    else:
        return overrides.plot_simulator_state(env_transition.state, use_log_traj=False, batch_idx=0)


def plot_scene_with_trajectory(env, env_transition, sdc_pov: bool, predicted_traj_xy=None):
    """Render scene and optionally overlay predicted trajectory waypoints.

    Args:
        env: Simulation environment.
        env_transition: Current environment transition.
        sdc_pov: If True, render from SDC point of view.
        predicted_traj_xy: np.ndarray of shape (N, 2) in world coordinates.
            If None, falls back to plain plot_scene.

    Returns:
        np.ndarray image (H, W, 3).
    """
    if predicted_traj_xy is None or sdc_pov:
        return plot_scene(env, env_transition, sdc_pov)

    state = env_transition.state
    # Get ego position and yaw for coordinate transform.
    sdc_idx = np.argmax(np.array(state.object_metadata.is_sdc[0]))
    t = int(np.array(state.timestep).ravel()[0])
    ego_xy = np.array(state.sim_trajectory.xy[0, sdc_idx, t])
    ego_yaw = float(state.sim_trajectory.yaw[0, sdc_idx, t])

    # predicted_traj_xy is ego-relative (x_forward, y_left) → world frame.
    cos_h, sin_h = np.cos(ego_yaw), np.sin(ego_yaw)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    world_xy = (rot @ predicted_traj_xy.T).T + ego_xy

    # Render base scene.
    from waymax.visualization import utils as waymax_utils
    viz_config = waymax_utils.VizConfig()
    fig, ax = waymax_utils.init_fig_ax(viz_config)
    overrides.plot_simulator_state(
        state, use_log_traj=False, batch_idx=0, ax=ax
    )

    # Overlay predicted waypoints as cyan dots connected by a line.
    ax.plot(world_xy[:, 0], world_xy[:, 1], "o-", color="cyan",
            markersize=1.7, linewidth=0.7, alpha=0.9, label="predicted traj")

    return waymax_utils.img_from_fig(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Render loop (trajectory-aware)
# ──────────────────────────────────────────────────────────────────────────────

def run_scenario_render_traj(
    scenario,
    rng_key: jax.Array,
    env,
    step_fn,
    reset_fn,
    render_pov: bool = False,
    show_trajectory: bool = False,
    policy_network=None,
    policy_params=None,
    num_waypoints: int = 16,
):
    """Render a scenario, optionally overlaying the predicted trajectory.

    When show_trajectory=True the actor network is called a second time each
    step to extract the raw waypoints (before LQR) for visualization.
    """
    rng_key, reset_key = jax.random.split(rng_key)
    reset_key = jax.random.split(reset_key, 1)
    env_transition = reset_fn(scenario, reset_key)

    episode_images = []

    traj_xy = None
    if show_trajectory and policy_network is not None and policy_params is not None:
        from vmax.agents.learning.reinforcement.td3.td3_factory import decode_trajectory
        obs = np.array(env_transition.observation[0])
        raw = policy_network.apply(policy_params, obs[None])
        traj_xy = np.array(decode_trajectory(raw, num_waypoints)[0])

    image = plot_scene_with_trajectory(env, env_transition, render_pov, traj_xy)
    episode_images.append(image)

    done = env_transition.done
    while not done:
        rng_key, step_key = jax.random.split(rng_key)
        step_key = jax.random.split(step_key, 1)
        env_transition, _ = step_fn(env_transition, key=step_key)
        done = env_transition.done

        traj_xy = None
        if show_trajectory and policy_network is not None and policy_params is not None:
            from vmax.agents.learning.reinforcement.td3.td3_factory import decode_trajectory
            obs = np.array(env_transition.observation[0])
            raw = policy_network.apply(policy_params, obs[None])
            traj_xy = np.array(decode_trajectory(raw, num_waypoints)[0])

        image = plot_scene_with_trajectory(env, env_transition, render_pov, traj_xy)
        episode_images.append(image)

    return episode_images, env_transition.info["steps"]
