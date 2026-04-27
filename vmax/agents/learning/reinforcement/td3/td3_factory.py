# Copyright 2025 Valeo.

"""Factory functions for the Twin Delayed DDPG (TD3) algorithm."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from vmax.agents import datatypes, networks
from vmax.agents.networks.lqr.jax_lqr import jax_lqr

# ──────────────────────────────────────────────────────────────────────
# Trajectory decoding constants
# ──────────────────────────────────────────────────────────────────────
_TRAJ_DT = 0.1     # waypoint timestep [s] — must match Waymax DT
_TRAJ_SCALE = 5.0  # actor tanh output (−1,1) → physical offset (−5,5) m
_TRAJ_SPEED = 5.0  # nominal forward speed for baseline [m/s]

# ──────────────────────────────────────────────────────────────────────
# Observation layout constants (must match base_config.yaml)
# Flat obs: [sdc(5steps×8feat), other_agents(8×5×8), rg, tl, path]
# Per-step SDC features: xy(2), vel_xy(2), yaw(1), length(1), width(1), valid(1)
# Current timestep (step 4): offset 4×8=32 → vel_xy at [34, 35]
# ──────────────────────────────────────────────────────────────────────
_MAX_OBS_SPEED = 30.0        # matches simulator/features/extractor/utils.py MAX_SPEED
_SDC_VEL_XY_IDX = slice(34, 36)


def decode_trajectory(raw: jnp.ndarray, num_waypoints: int) -> jnp.ndarray:
    """Map actor tanh output → ego-relative trajectory (metres).

    Baseline: waypoint i = ((i+1)*DT*speed, 0) — straight at nominal speed.
    Actor output is a scaled DELTA around that baseline so that
    zero-initialised weights produce a physically meaningful trajectory
    (straight line at 5 m/s) rather than a degenerate zero path.

    Args:
        raw:          (batch, num_waypoints*2) tanh-bounded actor output.
        num_waypoints: number of waypoints.

    Returns:
        trajectory: (batch, num_waypoints, 2) ego-relative [x_fwd, y_left] m.
    """
    t = jnp.arange(1, num_waypoints + 1) * _TRAJ_DT            # (N,)
    baseline = jnp.stack([t * _TRAJ_SPEED,
                          jnp.zeros(num_waypoints)], axis=-1)   # (N, 2)
    delta = raw.reshape(-1, num_waypoints, 2) * _TRAJ_SCALE     # (B, N, 2)
    return baseline[None] + delta                                # (B, N, 2)


@flax.struct.dataclass
class TD3NetworkParams:
    """Parameters for TD3 networks."""

    policy: datatypes.Params
    value: datatypes.Params
    target_policy: datatypes.Params
    target_value: datatypes.Params


@flax.struct.dataclass
class TD3Networks:
    """TD3 networks."""

    policy_network: Any
    value_network: Any
    policy_optimizer: Any
    value_optimizer: Any


@flax.struct.dataclass
class TD3TrainingState(datatypes.TrainingState):
    """Training state for TD3 algorithm."""

    params: TD3NetworkParams
    policy_optimizer_state: optax.OptState
    value_optimizer_state: optax.OptState
    rl_gradient_steps: int


def initialize(
    action_size: int,
    observation_size: int,
    env: Any,
    learning_rate: float,
    network_config: dict,
    num_devices: int,
    key: jax.Array,
    trajectory_size: int = 0,
) -> tuple[TD3Networks, TD3TrainingState, datatypes.Policy]:
    """Initialize TD3 components.

    Args:
        trajectory_size: When > 0, the actor outputs a flat trajectory of this
            size (e.g. 16*2=32).  LQR converts it to a 2-dim control before
            env.step; the critic still operates on the 2-dim action space.
            When 0 (default), actor directly outputs action_size controls.
    """
    network = make_networks(
        observation_size=observation_size,
        action_size=action_size,
        unflatten_fn=env.get_wrapper_attr(
            "features_extractor"
        ).unflatten_features,
        learning_rate=learning_rate,
        network_config=network_config,
        trajectory_size=trajectory_size,
    )

    policy_function = make_inference_fn(
        network, trajectory_size=trajectory_size
    )

    key_policy, key_value = jax.random.split(key)

    policy_params = network.policy_network.init(key_policy)
    policy_optimizer_state = network.policy_optimizer.init(policy_params)
    value_params = network.value_network.init(key_value)
    value_optimizer_state = network.value_optimizer.init(value_params)

    init_params = TD3NetworkParams(
        policy=policy_params,
        value=value_params,
        target_policy=policy_params,
        target_value=value_params,
    )

    training_state = TD3TrainingState(
        params=init_params,
        policy_optimizer_state=policy_optimizer_state,
        value_optimizer_state=value_optimizer_state,
        env_steps=0,
        rl_gradient_steps=0,
    )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:num_devices]
    )

    return network, training_state, policy_function


def make_inference_fn(
    td3_network: TD3Networks,
    trajectory_size: int = 0,
) -> datatypes.Policy:
    """Create the deterministic policy inference function for TD3.

    Args:
        td3_network: TD3 networks.
        trajectory_size: When > 0, the actor outputs a flat trajectory; the
            inference function applies JAX LQR to convert it to a 2-dim
            control action before returning.  Ego speed is estimated from the
            trajectory (distance to first waypoint / dt).
    """
    use_lqr = trajectory_size > 0
    num_waypoints = trajectory_size // 2 if use_lqr else 0

    def make_policy(
        params: datatypes.Params,
        deterministic: bool = False,
    ) -> datatypes.Policy:
        policy_network = td3_network.policy_network

        def policy(
            observations: jax.Array,
            key_sample: jax.Array = None,
        ) -> tuple[jax.Array, dict]:
            output = policy_network.apply(params, observations)

            if use_lqr:
                traj = decode_trajectory(output, num_waypoints)
                vel_xy_norm = observations[:, _SDC_VEL_XY_IDX]
                ego_speed = jnp.sqrt(jnp.sum(jnp.square(vel_xy_norm), axis=-1)) * _MAX_OBS_SPEED
                action = jax_lqr(traj, ego_speed)
            else:
                action = output

            if not deterministic and key_sample is not None:
                noise = (
                    jax.random.normal(key_sample, shape=action.shape) * 0.1
                )
                action = jnp.clip(action + noise, -1.0, 1.0)
            return action, {}

        return policy

    return make_policy


def make_networks(
    observation_size: int,
    action_size: int,
    unflatten_fn: callable,
    learning_rate: float,
    network_config: dict,
    trajectory_size: int = 0,
) -> TD3Networks:
    """Construct TD3 networks (deterministic actor + twin critics).

    Args:
        trajectory_size: When > 0, the actor outputs a flat trajectory of this
            size instead of a direct action.  The critic always takes the 2-dim
            control action (output of JAX LQR) as input.
    """
    actor_out_size = trajectory_size if trajectory_size > 0 else action_size
    policy_network = networks.make_policy_network(
        network_config, observation_size, actor_out_size, unflatten_fn
    )
    # Critic always evaluates 2-dim control actions
    value_network = networks.make_value_network(
        network_config, observation_size, action_size, unflatten_fn
    )

    policy_optimizer = optax.adam(learning_rate)
    value_optimizer = optax.adam(learning_rate)

    return TD3Networks(
        policy_network=policy_network,
        value_network=value_network,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
    )


def make_sgd_step(
    td3_network: TD3Networks,
    discount: float,
    tau: float,
    policy_freq: int,
    noise_std: float,
    noise_clip: float,
    trajectory_size: int = 0,
) -> datatypes.LearningFunction:
    """Create the SGD step function for TD3.

    Critic updated every step; actor and targets updated every
    policy_freq steps.
    """
    value_loss_fn, policy_loss_fn = _make_loss_fn(
        td3_network=td3_network,
        discount=discount,
        noise_std=noise_std,
        noise_clip=noise_clip,
        trajectory_size=trajectory_size,
    )

    value_update = networks.gradient_update_fn(
        value_loss_fn, td3_network.value_optimizer, pmap_axis_name="batch"
    )
    policy_update = networks.gradient_update_fn(
        policy_loss_fn, td3_network.policy_optimizer, pmap_axis_name="batch"
    )

    def sgd_step(
        carry: tuple[TD3TrainingState, jax.Array],
        transitions: datatypes.RLTransition,
    ) -> tuple[tuple[TD3TrainingState, jax.Array], datatypes.Metrics]:
        training_state, key = carry
        key, key_noise = jax.random.split(key)

        # --- Critic update (every step) ---
        value_loss, value_params, value_optimizer_state = value_update(
            training_state.params.value,
            training_state.params.target_policy,
            training_state.params.target_value,
            transitions,
            key_noise,
            optimizer_state=training_state.value_optimizer_state,
        )

        # --- Actor + target update (every policy_freq steps) ---
        do_policy_update = (
            training_state.rl_gradient_steps % policy_freq
        ) == 0

        def _update_policy(args):
            policy_params, opt_state = args
            loss, new_params, new_opt_state = policy_update(
                policy_params,
                value_params,
                transitions,
                optimizer_state=opt_state,
            )
            return loss, new_params, new_opt_state

        def _skip_policy(args):
            policy_params, opt_state = args
            return jnp.zeros(()), policy_params, opt_state

        policy_loss, policy_params, policy_optimizer_state = jax.lax.cond(
            do_policy_update,
            _update_policy,
            _skip_policy,
            (
                training_state.params.policy,
                training_state.policy_optimizer_state,
            ),
        )

        def _update_targets(args):
            new_value, new_policy = args
            target_value = jax.tree_util.tree_map(
                lambda t, n: t * (1 - tau) + n * tau,
                training_state.params.target_value, new_value,
            )
            target_policy = jax.tree_util.tree_map(
                lambda t, n: t * (1 - tau) + n * tau,
                training_state.params.target_policy, new_policy,
            )
            return target_value, target_policy

        def _skip_targets(args):
            return (
                training_state.params.target_value,
                training_state.params.target_policy,
            )

        target_value, target_policy = jax.lax.cond(
            do_policy_update,
            _update_targets,
            _skip_targets,
            (value_params, policy_params),
        )

        params = TD3NetworkParams(
            policy=policy_params,
            value=value_params,
            target_policy=target_policy,
            target_value=target_value,
        )

        training_state = training_state.replace(
            params=params,
            policy_optimizer_state=policy_optimizer_state,
            value_optimizer_state=value_optimizer_state,
            rl_gradient_steps=training_state.rl_gradient_steps + 1,
        )

        metrics = {"policy_loss": policy_loss, "value_loss": value_loss}
        return (training_state, key), metrics

    return sgd_step


def _make_loss_fn(
    td3_network: TD3Networks,
    discount: float,
    noise_std: float,
    noise_clip: float,
    trajectory_size: int = 0,
) -> tuple[callable, callable]:
    """Define TD3 loss functions.

    When trajectory_size > 0 (trajectory mode):
    - The actor outputs a flat trajectory (trajectory_size dims).
    - JAX LQR converts it to a 2-dim control action before passing to the
      critic.  Gradients flow through the LQR (pure JAX matrix ops).
    - The replay buffer stores 2-dim control actions, so the critic loss is
      identical to standard TD3.
    """
    policy_network = td3_network.policy_network
    value_network = td3_network.value_network
    use_lqr = trajectory_size > 0
    num_waypoints = trajectory_size // 2 if use_lqr else 0

    def _actor_to_action(params, obs):
        """Apply actor network and optionally convert via LQR."""
        output = policy_network.apply(params, obs)
        if use_lqr:
            traj = decode_trajectory(output, num_waypoints)
            vel_xy_norm = obs[:, _SDC_VEL_XY_IDX]
            ego_speed = jnp.sqrt(jnp.sum(jnp.square(vel_xy_norm), axis=-1)) * _MAX_OBS_SPEED
            return jax_lqr(traj, ego_speed)
        return output

    def compute_value_loss(
        value_params: datatypes.Params,
        target_policy_params: datatypes.Params,
        target_value_params: datatypes.Params,
        transitions: datatypes.RLTransition,
        key: jax.Array,
    ) -> jax.Array:
        # Target policy smoothing (noise on 2-dim control space)
        next_action = _actor_to_action(
            target_policy_params, transitions.next_observation
        )
        noise = jnp.clip(
            jax.random.normal(key, shape=next_action.shape) * noise_std,
            -noise_clip, noise_clip,
        )
        next_action = jnp.clip(next_action + noise, -1.0, 1.0)

        # Twin critic targets → take min
        next_q = value_network.apply(
            target_value_params, transitions.next_observation, next_action
        )
        target_q = (
            transitions.reward
            + transitions.flag * discount * jnp.min(next_q, axis=-1)
        )
        target_q = jax.lax.stop_gradient(target_q)

        # transitions.action is always the 2-dim control stored by policy_step
        q_values = value_network.apply(
            value_params, transitions.observation, transitions.action
        )
        value_loss = jnp.mean(
            jnp.sum(
                jnp.square(q_values - jnp.expand_dims(target_q, -1)), axis=-1
            )
        )
        return value_loss

    def compute_policy_loss(
        policy_params: datatypes.Params,
        value_params: datatypes.Params,
        transitions: datatypes.RLTransition,
    ) -> jax.Array:
        # Gradient flows: obs → actor → [LQR] → action → Q
        action = _actor_to_action(policy_params, transitions.observation)
        q_values = value_network.apply(
            value_params, transitions.observation, action
        )
        return -jnp.mean(q_values[..., 0])

    return compute_value_loss, compute_policy_loss
