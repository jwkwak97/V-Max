# Copyright 2025 Valeo.

"""JAX LQR controller: trajectory → normalized control action.

Architecture:
    trajectory (batch, N, 2) [ego-relative] + ego_speed (batch,)
    -> [acceleration, steering_curvature] normalized to [-1, 1]

Longitudinal 1-step LQR (velocity tracking):
    dynamics: v_next = v + a * dt
    cost: Q_lon * (v_next - v_ref)^2 + R_lon * a^2
    solution: a = -Q_lon*dt / (Q_lon*dt^2 + R_lon) * (v - v_ref)

Lateral 1-step LQR (heading tracking):
    state: [lateral_error, heading_error]
    input: dκ = κ - κ_ref
    dynamics:
        lat_err_next  = lat_err + v * heading_err * dt
        head_err_next = heading_err + dκ * v * dt
    cost: Q_head * head_err_next^2 + R_lat * dκ^2
    solution: dκ = -K * head_err

    Sign note: code's heading_err = arctan2(wp0_y, wp0_x) = θ_ref.
    In ego frame θ_ego = 0, so head_err = θ_ego - θ_ref = -heading_err.
    Therefore: dκ = -K * head_err = +K * heading_err.

Waymax InvertibleBicycleModel (normalize_actions=True):
    accel_phys = accel_norm * MAX_ACCEL  (MAX_ACCEL = 6.0 m/s^2)
    kappa_phys = kappa_norm * MAX_KAPPA  (MAX_KAPPA = 0.3 rad/m)
"""

import jax.numpy as jnp

# Physical limits matching Waymax InvertibleBicycleModel defaults
MAX_ACCEL = 6.0   # m/s^2
MAX_KAPPA = 0.3   # rad/m

# LQR weights — lateral gains kept small so computed kappa stays
# within [-MAX_KAPPA, MAX_KAPPA] for tanh-bounded actor outputs.
Q_LON = 10.0
R_LON = 1.0
Q_HEAD = 0.5
R_LAT = 20.0

# Waymax timestep
DT = 0.1  # seconds

# Safety floor to avoid division by zero at low speed
MIN_SPEED = 0.5  # m/s

# Stanley-style lateral correction gain; decays with 1/v for stability
K_LAT = 0.1


def compute_trajectory_references(trajectory: jnp.ndarray) -> tuple:
    """Extract reference speed, heading, curvature from waypoints.

    Args:
        trajectory: (batch, N, 2) ego-relative waypoints (x_forward,
            y_left).  N >= 3 required.  Waypoints spaced ~DT s apart.

    Returns:
        ref_speed:   (batch,) target speed [m/s], clipped [0, 20]
        heading_err: (batch,) θ_ref = angle of wp0 from ego [rad];
                     positive = reference is to the left
        lateral_err: (batch,) y-offset of wp0 [m]; positive = left
        ref_kappa:   (batch,) Menger curvature [rad/m], pre-clipped
    """
    wp0 = trajectory[:, 0, :]
    wp1 = trajectory[:, 1, :]
    wp2 = trajectory[:, 2, :]

    dist0 = jnp.linalg.norm(wp0, axis=-1)
    ref_speed = jnp.clip(dist0 / DT, 0.0, 20.0)

    heading_err = jnp.arctan2(wp0[:, 1], wp0[:, 0])
    lateral_err = wp0[:, 1]

    # Menger curvature: κ = 2*cross / (|d01| * |d12| * |d02|)
    d01 = wp1 - wp0
    d12 = wp2 - wp1
    d02 = wp2 - wp0
    cross = d01[:, 0] * d12[:, 1] - d01[:, 1] * d12[:, 0]
    len01 = jnp.linalg.norm(d01, axis=-1)
    len12 = jnp.linalg.norm(d12, axis=-1)
    len02 = jnp.linalg.norm(d02, axis=-1)
    denom = len01 * len12 * len02 + 1e-6
    ref_kappa = jnp.clip(
        2.0 * cross / denom,
        -MAX_KAPPA * 0.5,
        MAX_KAPPA * 0.5,
    )

    return ref_speed, heading_err, lateral_err, ref_kappa


def longitudinal_lqr(
    ego_speed: jnp.ndarray,
    ref_speed: jnp.ndarray,
) -> jnp.ndarray:
    """1-step LQR for longitudinal velocity tracking.

    Args:
        ego_speed: (batch,) current speed [m/s]
        ref_speed: (batch,) target speed [m/s]

    Returns:
        accel: (batch,) acceleration [m/s^2], clipped to ±MAX_ACCEL
    """
    gain = Q_LON * DT / (Q_LON * DT ** 2 + R_LON)
    accel = -gain * (ego_speed - ref_speed)
    return jnp.clip(accel, -MAX_ACCEL, MAX_ACCEL)


def lateral_lqr(
    ego_speed: jnp.ndarray,
    heading_err: jnp.ndarray,
    lateral_err: jnp.ndarray,
    ref_kappa: jnp.ndarray,
) -> jnp.ndarray:
    """1-step LQR for lateral/heading tracking.

    heading_err = θ_ref (angle of wp0 from ego origin), so
    head_err = θ_ego - θ_ref = -heading_err, giving
    dκ = -K * head_err = +K * heading_err.

    Args:
        ego_speed:   (batch,) current speed [m/s]
        heading_err: (batch,) θ_ref [rad]; positive = left
        lateral_err: (batch,) cross-track error [m]; positive = left
        ref_kappa:   (batch,) reference curvature [rad/m]

    Returns:
        kappa: (batch,) curvature command [rad/m], clipped to ±MAX_KAPPA
    """
    v = jnp.maximum(ego_speed, MIN_SPEED)
    vdt = v * DT
    gain_head = Q_HEAD * vdt / (Q_HEAD * vdt ** 2 + R_LAT)
    gain_lat = K_LAT / v

    dkappa = gain_head * heading_err + gain_lat * lateral_err
    kappa = ref_kappa + dkappa
    return jnp.clip(kappa, -MAX_KAPPA, MAX_KAPPA)


def jax_lqr(
    trajectory: jnp.ndarray,
    ego_speed: jnp.ndarray,
) -> jnp.ndarray:
    """Convert ego-relative trajectory + ego speed → control action.

    Fully differentiable (pure JAX).

    Args:
        trajectory: (batch, N, 2) ego-relative waypoints; N >= 3.
        ego_speed:  (batch,) current longitudinal speed [m/s].

    Returns:
        action: (batch, 2) [accel_norm, kappa_norm] in [-1, 1].
    """
    ref_speed, heading_err, lateral_err, ref_kappa = (
        compute_trajectory_references(trajectory)
    )

    accel = longitudinal_lqr(ego_speed, ref_speed)
    kappa = lateral_lqr(ego_speed, heading_err, lateral_err, ref_kappa)

    return jnp.stack([accel / MAX_ACCEL, kappa / MAX_KAPPA], axis=-1)
