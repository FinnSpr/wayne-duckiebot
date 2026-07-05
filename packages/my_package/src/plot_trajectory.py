#!/usr/bin/env python3
"""
Plot x-y trajectory from values0.csv comparing:
  1. Odometry (encoder-based pose estimate)
  2. Inferred trajectory (fitted model: v_act = k_v * v_cmd, ω_act = k_ω * ω_cmd)
  3. Ideal trajectory (theoretical: k=1, no losses)

Both old (6-column) and new (8-column with encoder deltas) CSV formats are supported.
The fitted gains are estimated from the data automatically.
"""

import csv
import math
import sys
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = "values0.csv"

# ----- Kinematic constants (must match test_trajectory.py) -----
R = 0.0318
BASELINE = 0.1


def load_csv(path: str):
    """
    Returns arrays: t, vl_cmd, vr_cmd, dphi_L, dphi_R, x_odo, y_odo, theta_odo
    Auto-detects old (6 cols) vs new (8 cols) format.
    """
    with open(path) as f:
        reader = csv.reader(f)
        first = next(reader)
        try:
            float(first[0])
            rows_raw = [first] + list(reader)
        except ValueError:
            rows_raw = list(reader)

    rows = [list(map(float, r)) for r in rows_raw]
    ncols = len(rows[0])

    t      = np.array([r[0] for r in rows])
    vl_cmd = np.array([r[1] for r in rows])
    vr_cmd = np.array([r[2] for r in rows])
    if ncols >= 8:
        dphi_L = np.array([r[3] for r in rows])
        dphi_R = np.array([r[4] for r in rows])
        x_odo  = np.array([r[5] for r in rows])
        y_odo  = np.array([r[6] for r in rows])
        th_odo = np.array([r[7] for r in rows])
    else:
        dphi_L = np.zeros(len(t))
        dphi_R = np.zeros(len(t))
        x_odo  = np.array([r[3] for r in rows])
        y_odo  = np.array([r[4] for r in rows])
        th_odo = np.array([r[5] for r in rows])
    return t, vl_cmd, vr_cmd, dphi_L, dphi_R, x_odo, y_odo, th_odo


def estimate_gains_from_odometry(t, x_odo, y_odo, th_odo, vl_cmd, vr_cmd):
    """
    Fallback: estimate k_v, k_ω by differencing the odometry and comparing
    to the _previous_ commanded speeds (accounts for one-step motor lag).
    Returns (k_v, k_ω).
    """
    dt = np.diff(t)
    dx = np.diff(x_odo)
    dy = np.diff(y_odo)
    dth = np.diff(th_odo)

    # Forward speed from odometry deltas
    v_odo = np.sqrt(dx**2 + dy**2) / dt
    w_odo = dth / dt

    # Aligned commands: v_odo[i] ~ from cmd[i] (approximate lag correction)
    v_cmd_avg = (vl_cmd[:-1] + vr_cmd[:-1]) / 2.0
    w_cmd_val = (vr_cmd[:-1] - vl_cmd[:-1]) / BASELINE

    # Skip startup transient (first ~2 seconds where odometry is erratic)
    start = 20
    mask_v = np.abs(v_cmd_avg[start:]) > 0.01
    mask_w = np.abs(w_cmd_val[start:]) > 0.01

    ratio_v = v_odo[start:][mask_v] / v_cmd_avg[start:][mask_v]
    ratio_w = w_odo[start:][mask_w] / w_cmd_val[start:][mask_w]

    # Robust: use median instead of mean (less sensitive to outliers)
    k_v = float(np.median(ratio_v)) if len(ratio_v) > 0 else 1.0
    k_w = float(np.median(ratio_w)) if len(ratio_w) > 0 else 1.0
    return k_v, k_w


def estimate_gains_from_encoders(t, vl_cmd, vr_cmd, dphi_L, dphi_R):
    """
    Preferred method: use raw encoder deltas properly aligned with commands.
    Returns (k_v, k_ω).
    """
    dt = np.diff(t)
    v_cmd_arr, v_act_arr, w_cmd_arr, w_act_arr = [], [], [], []

    for i in range(1, len(t)):
        if dt[i-1] <= 0:
            continue
        v_cmd_i = (vl_cmd[i-1] + vr_cmd[i-1]) / 2.0
        w_cmd_i = (vr_cmd[i-1] - vl_cmd[i-1]) / BASELINE
        v_act_i = R * (dphi_L[i] + dphi_R[i]) / (2.0 * dt[i-1])
        w_act_i = R * (dphi_R[i] - dphi_L[i]) / (BASELINE * dt[i-1])
        v_cmd_arr.append(v_cmd_i)
        v_act_arr.append(v_act_i)
        w_cmd_arr.append(w_cmd_i)
        w_act_arr.append(w_act_i)

    v_cmd_arr = np.array(v_cmd_arr)
    v_act_arr = np.array(v_act_arr)
    w_cmd_arr = np.array(w_cmd_arr)
    w_act_arr = np.array(w_act_arr)

    # Skip startup
    start = 15
    mask_v = np.abs(v_cmd_arr[start:]) > 0.01
    mask_w = np.abs(w_cmd_arr[start:]) > 0.01

    k_v = float(np.median(v_act_arr[start:][mask_v] / v_cmd_arr[start:][mask_v]))
    k_w = float(np.median(w_act_arr[start:][mask_w] / w_cmd_arr[start:][mask_w]))
    return k_v, k_w


def simulate_trajectory(t, vl_cmd, vr_cmd, k_v, k_w):
    """
    Integrate kinematics using the fitted gains.
    Returns (x_inf, y_inf, th_inf).
    """
    n = len(t)
    x, y, th = np.zeros(n), np.zeros(n), np.zeros(n)

    for i in range(1, n):
        dt_i = t[i] - t[i-1]
        if dt_i <= 0:
            x[i], y[i], th[i] = x[i-1], y[i-1], th[i-1]
            continue
        v = k_v * (vl_cmd[i-1] + vr_cmd[i-1]) / 2.0
        w = k_w * (vr_cmd[i-1] - vl_cmd[i-1]) / BASELINE
        th_mid = th[i-1] + w * dt_i / 2.0
        x[i] = x[i-1] + v * math.cos(th_mid) * dt_i
        y[i] = y[i-1] + v * math.sin(th_mid) * dt_i
        th[i] = th[i-1] + w * dt_i

    return x, y, th


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_FILE
    print(f"Loading {csv_path} ...")

    t, vl_cmd, vr_cmd, dphi_L, dphi_R, x_odo, y_odo, th_odo = load_csv(csv_path)
    print(f"  {len(t)} samples, duration {t[-1]:.1f}s")

    # ---- Estimate gains ----
    has_encoder_deltas = np.any(dphi_L != 0) or np.any(dphi_R != 0)
    if has_encoder_deltas:
        print("  Using encoder deltas for gain estimation (preferred)")
        k_v, k_w = estimate_gains_from_encoders(t, vl_cmd, vr_cmd, dphi_L, dphi_R)
    else:
        print("  No encoder deltas found – falling back to odometry differencing")
        k_v, k_w = estimate_gains_from_odometry(t, x_odo, y_odo, th_odo, vl_cmd, vr_cmd)

    print(f"  Fitted gains:  k_v = {k_v:.3f}   k_ω = {k_w:.3f}")

    # ---- Simulate inferred trajectory ----
    x_inf, y_inf, th_inf = simulate_trajectory(t, vl_cmd, vr_cmd, k_v, k_w)

    # ---- Simulate ideal trajectory (k=1, no losses) ----
    x_ideal, y_ideal, _ = simulate_trajectory(t, vl_cmd, vr_cmd, 1.0, 1.0)

    # ============================================================
    #  PLOT 1:  Side-by-side comparison (three panels)
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Panel A: Odometry ---
    ax = axes[0]
    sc0 = ax.scatter(x_odo, y_odo, c=t, cmap="viridis", s=12, edgecolor="none")
    ax.set_title("Odometry (encoder-based)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(sc0, ax=ax, label="Time (s)")

    # --- Panel B: Inferred (fitted model) ---
    ax = axes[1]
    sc1 = ax.scatter(x_inf, y_inf, c=t, cmap="plasma", s=12, edgecolor="none")
    ax.set_title(f"Inferred  (k_v={k_v:.2f}, k_ω={k_w:.2f})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(sc1, ax=ax, label="Time (s)")

    # --- Panel C: Ideal (k=1) ---
    ax = axes[2]
    sc2 = ax.scatter(x_ideal, y_ideal, c=t, cmap="cividis", s=12, edgecolor="none")
    ax.set_title("Ideal  (k_v=1.00, k_ω=1.00)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(sc2, ax=ax, label="Time (s)")

    fig.suptitle("Trajectory Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig("trajectory_comparison.png", dpi=150)
    print("Saved side-by-side comparison → trajectory_comparison.png")

    # ============================================================
    #  PLOT 2:  Overlay (same axes, different colormaps)
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(9, 8))

    # Odometry:  Viridis  (green → yellow)
    sc_odo = ax2.scatter(x_odo, y_odo, c=t, cmap="viridis", s=18,
                         edgecolor="none", label="Odometry", zorder=3)
    # Inferred:  Plasma  (purple → red → yellow)
    sc_inf = ax2.scatter(x_inf, y_inf, c=t, cmap="plasma", s=14,
                         edgecolor="none", label=f"Inferred (k_v={k_v:.2f}, k_ω={k_w:.2f})",
                         zorder=2)
    # Ideal:  Cool  (teal → blue)
    sc_ide = ax2.scatter(x_ideal, y_ideal, c=t, cmap="cool", s=10,
                         edgecolor="none", label="Ideal (k=1)", zorder=1,
                         alpha=0.7)

    ax2.set_xlabel("X position (m)")
    ax2.set_ylabel("Y position (m)")
    ax2.set_title("Overlaid Trajectories — Time-color-coded")
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=9)

    # Single shared colorbar using the odometry colormap as reference
    cbar = plt.colorbar(sc_odo, ax=ax2)
    cbar.set_label("Time (s)")

    plt.tight_layout()
    fig2.savefig("trajectory_overlay.png", dpi=150)
    print("Saved overlay plot           → trajectory_overlay.png")


if __name__ == "__main__":
    main()
