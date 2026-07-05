#!/usr/bin/env python3
"""
Plot commanded vs. actual wheel velocities over time.

Left wheel  → dashed lines
Right wheel → solid lines
Commands   → lighter shade
Actual     → darker shade  (computed from encoder delta_phi, or odometry fallback)
"""

import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE = "values0.csv"
R = 0.0318


def load_csv(path: str):
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


def actual_from_encoders(t, dphi_L, dphi_R):
    """Compute actual wheel velocities (m/s) from encoder deltas + dt."""
    dt = np.diff(t)
    vl_act = np.full(len(t), np.nan)
    vr_act = np.full(len(t), np.nan)
    for i in range(1, len(t)):
        if dt[i-1] > 0:
            vl_act[i] = R * dphi_L[i] / dt[i-1]
            vr_act[i] = R * dphi_R[i] / dt[i-1]
    return vl_act, vr_act


def actual_from_odometry(t, x_odo, y_odo, th_odo, vl_cmd, vr_cmd):
    """
    Fallback: back out actual wheel velocities from odometry pose deltas.
    Returns approximate vl_act, vr_act.
    """
    dt = np.diff(t)
    dx = np.diff(x_odo)
    dy = np.diff(y_odo)
    dth = np.diff(th_odo)

    vl_act = np.full(len(t), np.nan)
    vr_act = np.full(len(t), np.nan)

    for i in range(1, len(t)):
        if dt[i-1] <= 0:
            continue
        dist_center = np.sqrt(dx[i-1]**2 + dy[i-1]**2)
        # Sign from commanded direction
        sign = 1 if (vl_cmd[i-1] + vr_cmd[i-1]) >= 0 else -1
        dist_center *= sign
        dist_left  = dist_center - dth[i-1] * 0.1 / 2.0
        dist_right = dist_center + dth[i-1] * 0.1 / 2.0
        vl_act[i] = dist_left  / dt[i-1]
        vr_act[i] = dist_right / dt[i-1]

    return vl_act, vr_act


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else CSV_FILE
    print(f"Loading {csv_path} ...")
    t, vl_cmd, vr_cmd, dphi_L, dphi_R, x_odo, y_odo, th_odo = load_csv(csv_path)

    has_enc = np.any(dphi_L != 0) or np.any(dphi_R != 0)
    if has_enc:
        print("  Computing actual speeds from encoder deltas")
        vl_act, vr_act = actual_from_encoders(t, dphi_L, dphi_R)
    else:
        print("  No encoder deltas — falling back to odometry differencing")
        vl_act, vr_act = actual_from_odometry(t, x_odo, y_odo, th_odo, vl_cmd, vr_cmd)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 5))

    # Right wheel: solid lines
    # Left wheel:  dashed lines
    # Commands:    lighter / thinner
    # Actual:      darker  / thicker

    ax.plot(t, vl_cmd, linestyle="--", color="#74a9cf", linewidth=1.2, label="left cmd")
    ax.plot(t, vr_cmd, linestyle="-",  color="#fdae6b", linewidth=1.2, label="right cmd")
    ax.plot(t, vl_act, linestyle="--", color="#0570b0", linewidth=1.8, label="left actual")
    ax.plot(t, vr_act, linestyle="-",  color="#bd0026", linewidth=1.8, label="right actual")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wheel velocity (m/s)")
    ax.set_title("Commanded vs. Actual Wheel Velocities")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(left=0)

    plt.tight_layout()
    out = "velocity_plot.png"
    fig.savefig(out, dpi=150)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
