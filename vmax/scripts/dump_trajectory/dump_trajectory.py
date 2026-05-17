"""Dump per-step trajectory data for a trained model on one or more scenarios.

Captures:
  - ego (xy, yaw, vel_xy, speed)
  - path_target baseline (10 route points, ego-relative)
  - actor raw output (32 floats)
  - decoded predicted trajectory (16 waypoints, ego-relative)
  - residual = predicted_traj - baseline_interp
  - action (accel_norm, kappa_norm)
  - roadgraph_points (once per scenario)

Output: one pickle file per scenario, schema mirrors nuPlan's SimulationLog
(flat per-step dict list under a single scenario dict).

Usage:
  python vmax/scripts/dump_trajectory/dump_trajectory.py \
      --path_model "ritp_phase1_v6_ttc" \
      --path_dataset /path/to/data.tfrecord \
      --scenario_indexes 0 1 2 \
      --out_dir /home/jovyan/workspace/dump_results
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from vmax.scripts.evaluate_traj import utils
from vmax.scripts.training.train_utils import str2bool
from vmax.simulator import datasets, make_data_generator
from vmax.agents.learning.reinforcement.td3.td3_factory import (
    decode_trajectory,
    _NUM_PATH_POINTS,
    _PATH_TARGET_SIZE,
    _PATH_NORM_METERS,
)


def parse_args():
    p = argparse.ArgumentParser(description="Dump per-step trajectory data")
    p.add_argument("--path_model", type=str, required=True,
                   help="Run folder name under runs/ (e.g. ritp_phase1_v6_ttc)")
    p.add_argument("--path_dataset", type=str, required=True,
                   help="Path to TFRecord dataset")
    p.add_argument("--src_dir", type=str, default="runs",
                   help="Source directory containing run folders")
    p.add_argument("--out_dir", type=str,
                   default="/home/jovyan/workspace/dump_results",
                   help="Where to save the per-scenario pickle dumps")
    p.add_argument("--scenario_indexes", nargs="+", type=int, default=[0],
                   help="Scenario indexes to dump (0-based)")
    p.add_argument("--max_num_objects", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--waymo_dataset", type=str2bool, default=False)
    return p.parse_args()


def _to_np(x):
    return np.asarray(x)


def collect_step(env_transition, policy_network, policy_params, num_waypoints):
    """Capture all per-step data into a flat dict."""
    state = env_transition.state
    sdc_idx = int(np.argmax(_to_np(state.object_metadata.is_sdc[0])))
    t = int(_to_np(state.timestep).ravel()[0])

    obs = _to_np(env_transition.observation[0])           # (obs_size,)
    obs_batch = obs[None]                                  # (1, obs_size)

    # Actor raw + decoded trajectory
    raw = policy_network.apply(policy_params, obs_batch)
    raw_np = _to_np(raw)[0]                                # (32,)
    pred_traj = _to_np(
        decode_trajectory(raw, num_waypoints, obs_batch)
    )[0]                                                   # (16, 2)

    # Baseline path_target (ego-relative)
    # obs stores it NORMALIZED (÷ _PATH_NORM_METERS=50m); de-normalise to metres
    # to match the actual baseline used inside decode_trajectory.
    path_target_norm = obs[-_PATH_TARGET_SIZE:].reshape(_NUM_PATH_POINTS, 2)
    path_target = path_target_norm * _PATH_NORM_METERS     # metres, ego-relative

    # Interpolate baseline to 16 waypoints for residual comparison (in metres)
    t_src = np.linspace(0.0, 1.0, _NUM_PATH_POINTS)
    t_dst = np.linspace(0.0, 1.0, num_waypoints)
    baseline_interp = np.stack([
        np.interp(t_dst, t_src, path_target[:, 0]),
        np.interp(t_dst, t_src, path_target[:, 1]),
    ], axis=-1)                                            # (16, 2) metres
    residual = pred_traj - baseline_interp                  # (16, 2) metres

    # Ego state (world frame)
    ego_xy = _to_np(state.sim_trajectory.xy[0, sdc_idx, t])
    ego_yaw = float(_to_np(state.sim_trajectory.yaw[0, sdc_idx, t]))
    ego_vel_xy = _to_np(state.sim_trajectory.vel_xy[0, sdc_idx, t])
    ego_speed = float(np.linalg.norm(ego_vel_xy))

    return {
        "step": t,
        "timestamp_s": t * 0.1,
        "ego_xy": ego_xy,                  # (2,) world frame
        "ego_yaw": ego_yaw,                # rad
        "ego_vel_xy": ego_vel_xy,          # (2,) world frame
        "ego_speed": ego_speed,            # m/s
        "obs": obs,                        # full flat obs (for reproducibility)
        "actor_raw": raw_np,               # (32,)
        "predicted_traj": pred_traj,       # (16, 2) ego-relative
        "path_target": path_target,        # (10, 2) ego-relative normalized
        "baseline_interp": baseline_interp, # (16, 2) ego-relative
        "residual": residual,              # (16, 2) = pred_traj - baseline_interp
    }


def collect_roadgraph(env_transition):
    """Capture roadgraph once per scenario."""
    state = env_transition.state
    rg = state.roadgraph_points
    return {
        "xy": np.stack([_to_np(rg.x[0]), _to_np(rg.y[0])], axis=-1),  # (N_rg, 2)
        "dir_xy": np.stack([_to_np(rg.dir_x[0]), _to_np(rg.dir_y[0])], axis=-1),
        "types": _to_np(rg.types[0]),
        "ids": _to_np(rg.ids[0]),
        "valid": _to_np(rg.valid[0]),
    }


def dump_one_scenario(
    scenario,
    rng_key,
    env,
    step_fn,
    reset_fn,
    policy_network,
    policy_params,
    num_waypoints,
    scenario_idx,
):
    """Run one scenario to completion, capturing every step."""
    rng_key, reset_key = jax.random.split(rng_key)
    reset_key = jax.random.split(reset_key, 1)
    env_transition = reset_fn(scenario, reset_key)

    roadgraph = collect_roadgraph(env_transition)
    steps = [collect_step(env_transition, policy_network, policy_params, num_waypoints)]

    # Record action AFTER each step (action that took us from t→t+1)
    actions = []

    done = bool(_to_np(env_transition.done).ravel()[0])
    while not done:
        rng_key, step_key = jax.random.split(rng_key)
        step_key = jax.random.split(step_key, 1)
        env_transition, info = step_fn(env_transition, key=step_key)

        # action taken in the previous step is in env_transition's info if available
        # otherwise re-derive from current obs (deterministic eval mode)
        prev_obs = _to_np(env_transition.observation[0])[None]
        # NOTE: this re-derived action is the action that WOULD be taken at the
        # new state, not the one that transitioned us here. For analysis we keep
        # both: actor output at each visited state.

        done = bool(_to_np(env_transition.done).ravel()[0])
        if not done:
            steps.append(collect_step(env_transition, policy_network, policy_params, num_waypoints))

    return {
        "scenario_idx": int(scenario_idx),
        "dt": 0.1,
        "num_waypoints": num_waypoints,
        "num_path_points": _NUM_PATH_POINTS,
        "roadgraph": roadgraph,
        "steps": steps,
    }


def main():
    args = parse_args()

    print(f"-> Loading model: {args.path_model}")
    env, step_fn, eval_path, _, trajectory_size = utils.setup_evaluation(
        policy_type="ai",
        path_model=args.path_model,
        source_dir=args.src_dir,
        path_dataset=args.path_dataset,
        eval_name="/tmp/_dump_trajectory_eval",  # placeholder, not used
        max_num_objects=args.max_num_objects,
        noisy_init=False,
        sdc_paths_from_data=not args.waymo_dataset,
    )
    num_waypoints = trajectory_size // 2 if trajectory_size > 0 else 16
    print(f"-> trajectory_size={trajectory_size}, num_waypoints={num_waypoints}")

    # Build the raw policy network + load params (for direct apply)
    from vmax.agents.learning.reinforcement.td3.td3_factory import make_networks
    from vmax.scripts.evaluate_traj.utils import (
        get_model_path, load_params, load_yaml_config,
    )

    run_path = f"{args.src_dir}/{args.path_model}/"
    model_path, _ = get_model_path(run_path + "model/")
    params = load_params(model_path)
    policy_params = params.policy

    cfg = load_yaml_config(run_path + ".hydra/config.yaml")
    cfg["encoder"] = cfg["network"]["encoder"]
    cfg["policy"] = cfg["algorithm"]["network"]["policy"]
    cfg["value"] = cfg["algorithm"]["network"].get("value")
    cfg["unflatten_config"] = cfg["observation_config"]
    cfg["action_distribution"] = cfg["algorithm"]["network"].get("action_distribution")
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features
    network = make_networks(
        observation_size=env.observation_spec(),
        action_size=env.action_spec().data.shape[0],
        unflatten_fn=unflatten_fn,
        learning_rate=cfg["algorithm"]["learning_rate"],
        network_config=cfg,
        trajectory_size=trajectory_size,
    )
    policy_network = network.policy_network

    # Data generator
    batch_dims = (1,)   # one scenario at a time
    include_sdc_paths = not args.waymo_dataset
    data_generator = make_data_generator(
        path=datasets.get_dataset(args.path_dataset),
        max_num_objects=args.max_num_objects,
        include_sdc_paths=include_sdc_paths,
        batch_dims=batch_dims,
        seed=args.seed,
        repeat=1,
    )

    jitted_step_fn = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)

    out_dir = Path(args.out_dir) / args.path_model
    out_dir.mkdir(parents=True, exist_ok=True)

    target_indexes = set(args.scenario_indexes)
    rng_key = jax.random.PRNGKey(args.seed)

    for idx, scenario in enumerate(data_generator):
        if idx not in target_indexes:
            continue

        print(f"-> Dumping scenario {idx} ...")
        rng_key, sk = jax.random.split(rng_key)
        result = dump_one_scenario(
            scenario, sk, env, jitted_step_fn, jitted_reset,
            policy_network, policy_params, num_waypoints, idx,
        )

        out_path = out_dir / f"scenario_{idx:04d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_steps = len(result["steps"])
        print(f"   → {n_steps} steps  saved: {out_path}")

        if idx >= max(target_indexes):
            break

    print(f"\n✅ All dumps written to: {out_dir}")


if __name__ == "__main__":
    main()
