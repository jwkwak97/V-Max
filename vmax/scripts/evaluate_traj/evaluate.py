"""Evaluation script for trajectory-output models (td3_trajectory mode).

Differences from evaluate/evaluate.py:
  - Imports utils from evaluate_traj.utils (trajectory_size-aware load_model).
  - setup_evaluation also returns trajectory_size.
  - Adds --show_trajectory flag: when rendering, overlays predicted 16-waypoint
    trajectory on each frame as cyan dots.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import time
from functools import partial

import jax
import numpy as np
from tqdm import tqdm

from vmax.scripts.evaluate_traj import utils
from vmax.scripts.evaluate.utils import run_scenario_jit, write_video
from vmax.scripts.training.train_utils import str2bool
from vmax.simulator import datasets, make_data_generator


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Trajectory model evaluation arguments")
    parser.add_argument("--sdc_actor", "-sdc", type=str, default="expert",
                        help="Actor type: 'ai' for learned policy (default: expert)")
    parser.add_argument("--max_num_objects", "-o", type=int, default=64,
                        help="Maximum number of objects in the scene (default: 64)")
    parser.add_argument("--scenario_indexes", "-si", nargs="*", type=int, default=None,
                        help="Optional list of scenario indexes to evaluate")
    parser.add_argument("--render", "-r", type=str2bool, default=False,
                        help="Render mp4 — full scene view (default: False)")
    parser.add_argument("--sdc_pov", "-pov", type=str2bool, default=False,
                        help="Render mp4 — SDC point-of-view (default: False)")
    parser.add_argument("--show_trajectory", "-st", type=str2bool, default=False,
                        help="Overlay predicted waypoints on rendered frames (default: False). "
                             "Only active when --render is True.")
    parser.add_argument("--path_dataset", "-pd", type=str, default="local_womd_valid",
                        help="Path to the TFRecord dataset")
    parser.add_argument("--path_model", "-pm", type=str, default="",
                        help="Run folder name under runs/ (e.g. TD3_VEC_WAYFORMER_24-04_05:19:56)")
    parser.add_argument("--eval_name", "-en", type=str, default="benchmark",
                        help="Base directory for saving evaluation results (default: benchmark)")
    parser.add_argument("--noisy_init", "-ni", type=str2bool, default=False,
                        help="Enable noisy initialisation (default: False)")
    parser.add_argument("--src_dir", "-sd", type=str, default="runs",
                        help="Source directory containing run folders (default: runs)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--batch_size", "-bs", type=int, default=1,
                        help="Parallel scenarios (must be 1 when --render/--sdc_pov, default: 1)")
    parser.add_argument("--waymo_dataset", "-wd", type=str2bool, default=False,
                        help="Use Waymo dataset format (default: False)")

    args = parser.parse_args()

    if (args.render or args.sdc_pov) and args.batch_size > 1:
        raise ValueError("batch_size must be 1 when --render or --sdc_pov is set.")
    if args.sdc_actor == "ai" and args.path_model == "":
        raise ValueError("--path_model must be provided when --sdc_actor ai.")
    if args.show_trajectory and not (args.render or args.sdc_pov):
        print("Warning: --show_trajectory has no effect without --render or --sdc_pov.")

    return args


def run_evaluation(
    env,
    data_generator,
    step_fn,
    run_path: str = "",
    scenario_indexes=None,
    termination_keys=None,
    render: bool = False,
    render_pov: bool = False,
    show_trajectory: bool = False,
    trajectory_size: int = 0,
    policy_network=None,
    policy_params=None,
    seed: int = 0,
    batch_size: int = 1,
):
    jitted_step_fn = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)

    rng_key = jax.random.PRNGKey(seed)
    eval_metrics = {"episode_length": [], "accuracy": []}
    start_time_total = time.time()

    progress_bar = tqdm(desc="Evaluating scenarios", unit=" scenario")
    rendering = render or render_pov
    num_waypoints = trajectory_size // 2 if trajectory_size > 0 else 16

    if rendering:
        if show_trajectory and policy_network is not None:
            _run_scenario = partial(
                utils.run_scenario_render_traj,
                env=env,
                step_fn=jitted_step_fn,
                reset_fn=jitted_reset,
                render_pov=render_pov,
                show_trajectory=True,
                policy_network=policy_network,
                policy_params=policy_params,
                num_waypoints=num_waypoints,
            )
        else:
            _run_scenario = partial(
                utils.run_scenario_render_traj,
                env=env,
                step_fn=jitted_step_fn,
                reset_fn=jitted_reset,
                render_pov=render_pov,
                show_trajectory=False,
            )
    else:
        _run_scenario = partial(run_scenario_jit, step_fn=jitted_step_fn, reset_fn=jitted_reset)
        _run_scenario = jax.vmap(_run_scenario)
        _run_scenario = jax.jit(_run_scenario)

    total_scenarios = 0
    for scenario in data_generator:
        if scenario_indexes is not None and total_scenarios not in scenario_indexes:
            total_scenarios += batch_size
            continue

        rng_key, scenario_key = jax.random.split(rng_key)

        if rendering:
            images, steps_done = _run_scenario(scenario, scenario_key)
            write_video(run_path, images, total_scenarios)
            images.clear()
        else:
            scenario_key = jax.random.split(scenario_key, batch_size)
            episode_metrics, steps_done = _run_scenario(scenario, scenario_key)
            eval_metrics = utils.append_episode_metrics(
                steps_done, eval_metrics, episode_metrics, termination_keys, batch_size
            )

        progress_bar.update(batch_size)
        total_scenarios += batch_size

    progress_bar.close()
    total_time = time.time() - start_time_total

    if not rendering:
        utils.write_generator_result(run_path, total_scenarios, eval_metrics)

    print(
        f"-> Evaluation completed: {total_scenarios} episodes in {total_time:.2f}s "
        f"(avg {total_time / total_scenarios:.2f}s per episode)"
    )
    return None if rendering else eval_metrics


def main():
    eval_args = parse_eval_args()
    print(f"-> Setting up evaluation for {eval_args.sdc_actor} policy on {eval_args.path_dataset} dataset...")

    batch_dims = (1,) if (eval_args.render or eval_args.sdc_pov) else (eval_args.batch_size, 1)
    include_sdc_paths = not eval_args.waymo_dataset

    data_generator = make_data_generator(
        path=datasets.get_dataset(eval_args.path_dataset),
        max_num_objects=eval_args.max_num_objects,
        include_sdc_paths=include_sdc_paths,
        batch_dims=batch_dims,
        seed=eval_args.seed,
        repeat=1,
    )

    env, step_fn, eval_path, termination_keys, trajectory_size = utils.setup_evaluation(
        eval_args.sdc_actor,
        eval_args.path_model,
        eval_args.src_dir,
        eval_args.path_dataset,
        eval_args.eval_name,
        eval_args.max_num_objects,
        eval_args.noisy_init,
        include_sdc_paths,
    )

    print(f"-> trajectory_size: {trajectory_size}")
    print(f"-> Starting evaluation with output path: {eval_path}")

    # For trajectory overlay: extract raw actor network and params from a
    # fresh build so we can call apply() independently per step.
    policy_network = None
    policy_params = None
    if eval_args.show_trajectory and trajectory_size > 0 and eval_args.sdc_actor == "ai":
        try:
            from vmax.scripts.evaluate_traj.utils import (
                get_algorithm_modules,
                load_params,
                load_yaml_config,
                get_model_path,
            )
            run_path_full = f"{eval_args.src_dir}/{eval_args.path_model}/"
            model_path, _ = get_model_path(run_path_full + "model/")
            params = load_params(model_path)
            policy_params = params.policy
            # Use the policy_network from the environment-built network
            from vmax.agents.learning.reinforcement.td3.td3_factory import make_networks
            from vmax.scripts.evaluate_traj.utils import load_yaml_config
            eval_config = load_yaml_config(run_path_full + ".hydra/config.yaml")
            eval_config["encoder"] = eval_config["network"]["encoder"]
            eval_config["policy"] = eval_config["algorithm"]["network"]["policy"]
            eval_config["value"] = eval_config["algorithm"]["network"].get("value")
            eval_config["unflatten_config"] = eval_config["observation_config"]
            eval_config["action_distribution"] = eval_config["algorithm"]["network"].get("action_distribution")
            unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features
            network = make_networks(
                observation_size=env.observation_spec(),
                action_size=env.action_spec().data.shape[0],
                unflatten_fn=unflatten_fn,
                learning_rate=eval_config["algorithm"]["learning_rate"],
                network_config=eval_config,
                trajectory_size=trajectory_size,
            )
            policy_network = network.policy_network
        except Exception as e:
            import traceback
            print(f"Warning: could not set up trajectory overlay ({e}).")
            traceback.print_exc()
            print("Continuing without overlay.")

    print(f"[DEBUG] show_trajectory={eval_args.show_trajectory}, policy_network={'set' if policy_network is not None else 'None'}, policy_params={'set' if policy_params is not None else 'None'}")

    eval_metrics = run_evaluation(
        env,
        data_generator,
        step_fn,
        eval_path,
        eval_args.scenario_indexes,
        termination_keys,
        eval_args.render,
        eval_args.sdc_pov,
        eval_args.show_trajectory,
        trajectory_size,
        policy_network,
        policy_params,
        eval_args.seed,
        eval_args.batch_size,
    )

    if eval_metrics is not None:
        print("\n-> Evaluation Summary:")
        print(f"Accuracy: {np.mean(eval_metrics['accuracy']):.4f}")
        if "vmax_aggregate_score" in eval_metrics:
            print(f"V-Max Score: {np.mean(eval_metrics['vmax_aggregate_score']):.4f}")


if __name__ == "__main__":
    main()
