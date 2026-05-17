"""Analyze trajectory dumps: LQR tracking error + roadgraph/baseline/predicted_traj visualization.

Reads pickle files written by dump_trajectory.py and produces:
  1. LQR tracking error stats (text):
       error[t] = || ego_xy[t+1] - world_frame(predicted_traj[t][0]) ||
     i.e. how far the ego is from where the policy's first waypoint said it would be
     after 0.1 s.
  2. Per-scenario PNG overview:
       - roadgraph (light gray)
       - ego trail full episode (red)
       - at every K steps: path_target baseline (green) + predicted_traj (cyan)
                          + LQR predicted ego next pos (orange ×)
                          + actual next ego (red ●)

Usage:
  python vmax/scripts/dump_trajectory/analyze_dump.py \
      --dump_dir /home/jovyan/workspace/dump_results/ritp_phase1_v6_ttc \
      --overlay_every 2
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Analyze trajectory dumps")
    p.add_argument("--dump_dir", type=str, required=True,
                   help="Directory containing scenario_XXXX.pkl files")
    p.add_argument("--overlay_every", type=int, default=2,
                   help="Overlay baseline + predicted_traj every N steps in the plot")
    p.add_argument("--out_subdir", type=str, default="analysis",
                   help="Subdirectory under dump_dir for outputs")
    return p.parse_args()


def world_frame(traj_ego_rel: np.ndarray, ego_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    """Convert ego-relative (x_forward, y_left) trajectory to world frame.

    Args:
        traj_ego_rel: (N, 2) ego-relative metres
        ego_xy:       (2,)   ego world position
        ego_yaw:      scalar ego heading [rad]
    Returns:
        (N, 2) world-frame metres
    """
    cos_h, sin_h = np.cos(ego_yaw), np.sin(ego_yaw)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (rot @ traj_ego_rel.T).T + ego_xy


def compute_lqr_tracking_error(steps):
    """For each step t with a t+1 available, compute distance between
    actual ego[t+1] and predicted_traj[t][0] (world frame).

    Returns:
        errors: (n-1,) metres
        details: list of dicts with ego[t+1], predicted_next_world, error
    """
    errors = []
    details = []
    for t in range(len(steps) - 1):
        st = steps[t]
        st_next = steps[t + 1]

        # First waypoint of predicted trajectory at time t (ego-relative)
        first_wp_rel = st["predicted_traj"][0:1]            # (1, 2)
        first_wp_world = world_frame(first_wp_rel,
                                     st["ego_xy"], st["ego_yaw"])[0]
        actual_next = st_next["ego_xy"]
        err = float(np.linalg.norm(actual_next - first_wp_world))
        errors.append(err)
        details.append({
            "step": int(st["step"]),
            "predicted_next_world": first_wp_world,
            "actual_next": actual_next,
            "error": err,
        })
    return np.array(errors), details


def plot_scenario(d, out_path, overlay_every=2):
    """Render a top-down overview of one scenario."""
    rg = d["roadgraph"]
    valid = rg["valid"].astype(bool)
    rg_xy = rg["xy"][valid]

    steps = d["steps"]
    ego_xy_all = np.stack([s["ego_xy"] for s in steps])      # (n, 2)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Roadgraph background
    ax.scatter(rg_xy[:, 0], rg_xy[:, 1], s=0.5, c="lightgray",
               alpha=0.6, label="roadgraph")

    # Ego trail (full episode)
    ax.plot(ego_xy_all[:, 0], ego_xy_all[:, 1], "-", color="red",
            linewidth=1.5, alpha=0.8, label="ego trail")
    ax.scatter(ego_xy_all[0, 0], ego_xy_all[0, 1], s=80, c="red",
               marker="o", edgecolor="black", zorder=5, label="ego start")
    ax.scatter(ego_xy_all[-1, 0], ego_xy_all[-1, 1], s=80, c="darkred",
               marker="s", edgecolor="black", zorder=5, label="ego end")

    # Overlays at sampled steps
    first_label = True
    for t in range(0, len(steps), overlay_every):
        s = steps[t]
        # path_target baseline in world frame
        pt_world = world_frame(s["path_target"], s["ego_xy"], s["ego_yaw"])
        ax.plot(pt_world[:, 0], pt_world[:, 1], "o-", color="green",
                markersize=2, linewidth=0.8, alpha=0.5,
                label="path_target" if first_label else None)

        # predicted trajectory in world frame
        pred_world = world_frame(s["predicted_traj"], s["ego_xy"], s["ego_yaw"])
        ax.plot(pred_world[:, 0], pred_world[:, 1], "o-", color="cyan",
                markersize=2, linewidth=0.8, alpha=0.7,
                label="predicted_traj" if first_label else None)

        # predicted next ego (first waypoint) — orange ×
        ax.scatter(pred_world[0, 0], pred_world[0, 1], s=40,
                   c="orange", marker="x", linewidth=1.5,
                   label="predicted next ego" if first_label else None)

        first_label = False

    # Zoom around ego trail
    pad = 30.0
    ax.set_xlim(ego_xy_all[:, 0].min() - pad, ego_xy_all[:, 0].max() + pad)
    ax.set_ylim(ego_xy_all[:, 1].min() - pad, ego_xy_all[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"Scenario {d['scenario_idx']}: roadgraph + ego trail + "
        f"path_target/predicted_traj every {overlay_every} steps"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    dump_dir = Path(args.dump_dir)
    out_dir = dump_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []

    for pkl_path in sorted(dump_dir.glob("scenario_*.pkl")):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)

        idx = d["scenario_idx"]
        steps = d["steps"]
        errors, _ = compute_lqr_tracking_error(steps)

        line = (f"Scenario {idx:3d}  ({len(steps):3d} steps)  "
                f"LQR tracking err [m]: "
                f"mean={errors.mean():.3f}  "
                f"median={np.median(errors):.3f}  "
                f"max={errors.max():.3f}  "
                f"p95={np.percentile(errors, 95):.3f}")
        print(line)
        summary_lines.append(line)

        out_png = out_dir / f"scenario_{idx:04d}.png"
        plot_scenario(d, out_png, overlay_every=args.overlay_every)
        print(f"   → {out_png}")

    # Write summary
    summary_path = out_dir / "lqr_tracking_summary.txt"
    with open(summary_path, "w") as f:
        f.write("LQR Tracking Error Summary\n")
        f.write("(distance between predicted_traj[0] and actual ego next position)\n")
        f.write("=" * 70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"\n✅ Summary: {summary_path}")
    print(f"✅ Plots in: {out_dir}/")


if __name__ == "__main__":
    main()
