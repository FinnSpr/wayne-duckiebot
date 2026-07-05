#!/usr/bin/env python3
"""
Fit a motor transfer function:  v_actual = f(v_cmd)

Reads values0.csv (must contain encoder deltas — see updated test_trajectory.py),
aligns commands with their resulting encoder measurements, then fits several
candidate models.

Usage:
    python3 fit_motor_model.py [csv_file] [--plot]
"""

import csv
import sys
import math
import numpy as np
from typing import List, Tuple

# ----- Kinematic constants (must match test_trajectory.py) -----
R = 0.0318       # wheel radius [m]
BASELINE = 0.1   # distance between wheels [m]


def load_data(path: str) -> Tuple[np.ndarray, ...]:
    """
    Read CSV. Auto-detects format:
      OLD (6 cols): t, vel_left_cmd, vel_right_cmd, x_odo, y_odo, theta_odo
      NEW (8 cols): t, vel_left_cmd, vel_right_cmd, delta_phi_L, delta_phi_R, x_odo, y_odo, theta_odo

    Returns arrays: t, vl_cmd, vr_cmd, dphi_L, dphi_R, x_odo, y_odo, theta_odo
    (dphi_L, dphi_R are zeros for old format)
    """
    with open(path) as f:
        reader = csv.reader(f)
        first = next(reader)
        # Check if first row is a header or data
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
        # Old format: t, vl, vr, x, y, theta
        dphi_L = np.zeros(len(t))
        dphi_R = np.zeros(len(t))
        x_odo  = np.array([r[3] for r in rows])
        y_odo  = np.array([r[4] for r in rows])
        th_odo = np.array([r[5] for r in rows])
    return t, vl_cmd, vr_cmd, dphi_L, dphi_R, x_odo, y_odo, th_odo


def align_commands_to_encoders(
    t: np.ndarray, vl_cmd: np.ndarray, vr_cmd: np.ndarray,
    dphi_L: np.ndarray, dphi_R: np.ndarray,
    skip: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The encoder delta at row i was accumulated from the commands sent at row i-1.
    This function pairs each encoder measurement with its *causing* command.

    Returns (v_cmd, v_actual, omega_cmd, omega_actual) where:
        v_cmd     = (vl_cmd[i-1] + vr_cmd[i-1]) / 2
        v_actual  = R * (dphi_L[i] + dphi_R[i]) / (2 * dt[i])
        omega_cmd   = (vr_cmd[i-1] - vl_cmd[i-1]) / BASELINE
        omega_actual = R * (dphi_R[i] - dphi_L[i]) / (BASELINE * dt[i])
    """
    dt = np.diff(t)
    # We start from index `skip` to avoid encoder startup transients
    v_cmd_list, v_act_list, w_cmd_list, w_act_list = [], [], [], []

    for i in range(max(1, skip), len(t)):
        if dt[i-1] <= 0:
            continue
        # Command from previous step
        v_cmd_i = (vl_cmd[i-1] + vr_cmd[i-1]) / 2.0
        w_cmd_i = (vr_cmd[i-1] - vl_cmd[i-1]) / BASELINE
        # Actual from encoder deltas at current step
        v_act_i = R * (dphi_L[i] + dphi_R[i]) / (2.0 * dt[i-1])
        w_act_i = R * (dphi_R[i] - dphi_L[i]) / (BASELINE * dt[i-1])

        v_cmd_list.append(v_cmd_i)
        v_act_list.append(v_act_i)
        w_cmd_list.append(w_cmd_i)
        w_act_list.append(w_act_i)

    return (
        np.array(v_cmd_list), np.array(v_act_list),
        np.array(w_cmd_list), np.array(w_act_list),
    )


def fit_linear(v_cmd: np.ndarray, v_act: np.ndarray) -> dict:
    """Fit v_actual = k * v_cmd  (zero-intercept linear)."""
    # Least squares: k = sum(v_cmd * v_act) / sum(v_cmd^2)
    k = np.sum(v_cmd * v_act) / np.sum(v_cmd ** 2)
    residuals = v_act - k * v_cmd
    rmse = np.sqrt(np.mean(residuals ** 2))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((v_act - np.mean(v_act)) ** 2)
    return {"name": "linear (v_act = k * v_cmd)", "k": k, "rmse": rmse, "r2": r2}


def fit_affine(v_cmd: np.ndarray, v_act: np.ndarray) -> dict:
    """Fit v_actual = k * v_cmd + b  (affine with intercept)."""
    A = np.column_stack([v_cmd, np.ones_like(v_cmd)])
    k, b = np.linalg.lstsq(A, v_act, rcond=None)[0]
    residuals = v_act - (k * v_cmd + b)
    rmse = np.sqrt(np.mean(residuals ** 2))
    r2 = 1 - np.sum(residuals ** 2) / np.sum((v_act - np.mean(v_act)) ** 2)
    return {"name": "affine (v_act = k*v_cmd + b)", "k": k, "b": b, "rmse": rmse, "r2": r2}


def fit_deadzone(v_cmd: np.ndarray, v_act: np.ndarray) -> dict:
    """
    Fit v_actual = k * (v_cmd - d) for v_cmd > d
         v_actual = k * (v_cmd + d) for v_cmd < -d
         v_actual = 0 otherwise

    Uses grid search for d, then least-squares for k.
    """
    best_d, best_k, best_rmse = 0.0, 0.0, float("inf")

    for d in np.linspace(0, 0.15, 100):
        mask = np.abs(v_cmd) > d
        if np.sum(mask) < 10:
            continue
        # Effective command after dead zone
        v_eff = np.where(v_cmd > d, v_cmd - d, np.where(v_cmd < -d, v_cmd + d, 0.0))
        k = np.sum(v_eff[mask] * v_act[mask]) / np.sum(v_eff[mask] ** 2)
        residuals = v_act[mask] - k * v_eff[mask]
        rmse = np.sqrt(np.mean(residuals ** 2))
        if rmse < best_rmse:
            best_rmse, best_k, best_d = rmse, k, d

    v_eff_all = np.where(v_cmd > best_d, v_cmd - best_d,
                         np.where(v_cmd < -best_d, v_cmd + best_d, 0.0))
    residuals = v_act - best_k * v_eff_all
    r2 = 1 - np.sum(residuals ** 2) / np.sum((v_act - np.mean(v_act)) ** 2)
    return {"name": "dead-zone", "k": best_k, "d": best_d, "rmse": best_rmse, "r2": r2}


def simulate_trajectory(
    vl_cmd_seq: np.ndarray,
    vr_cmd_seq: np.ndarray,
    dt: float,
    model_v: callable,
    model_w: callable,
    x0: float = 0.0,
    y0: float = 0.0,
    th0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a sequence of commanded wheel speeds and models for v_actual, omega_actual,
    integrate the kinematic equations to produce a trajectory.

    model_v(vl_cmd, vr_cmd) -> v_actual
    model_w(vl_cmd, vr_cmd) -> omega_actual
    """
    n = len(vl_cmd_seq)
    x, y, th = np.zeros(n), np.zeros(n), np.zeros(n)
    x[0], y[0], th[0] = x0, y0, th0

    for i in range(1, n):
        v = model_v(vl_cmd_seq[i-1], vr_cmd_seq[i-1])
        w = model_w(vl_cmd_seq[i-1], vr_cmd_seq[i-1])
        # Midpoint integration
        th_mid = th[i-1] + w * dt / 2.0
        x[i] = x[i-1] + v * math.cos(th_mid) * dt
        y[i] = y[i-1] + v * math.sin(th_mid) * dt
        th[i] = th[i-1] + w * dt

    return x, y, th


# ---------------------------------------------------------------------------
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "values0.csv"
    do_plot = "--plot" in sys.argv

    print(f"Loading {csv_path} ...")
    t, vl_cmd, vr_cmd, dphi_L, dphi_R, x_odo, y_odo, th_odo = load_data(csv_path)
    print(f"  {len(t)} samples, duration {t[-1]:.1f}s")

    # ---- Step 1: Align commands with resulting encoder measurements ----
    v_cmd, v_act, w_cmd, w_act = align_commands_to_encoders(
        t, vl_cmd, vr_cmd, dphi_L, dphi_R, skip=15
    )
    print(f"\n  Aligned pairs: {len(v_cmd)}")

    # ---- Step 2: Fit candidate models ----
    print("\n========== Forward speed models ==========")
    for fit_fn in [fit_linear, fit_affine, fit_deadzone]:
        result = fit_fn(v_cmd, v_act)
        print(f"  {result['name']}:")
        for key, val in result.items():
            if key != "name":
                print(f"    {key} = {val:.4f}")
        print()

    print("\n========== Angular velocity models ==========")
    for fit_fn in [fit_linear, fit_affine, fit_deadzone]:
        result = fit_fn(w_cmd, w_act)
        print(f"  {result['name']}:")
        for key, val in result.items():
            if key != "name":
                print(f"    {key} = {val:.4f}")
        print()

    # ---- Step 3: If old-format CSV, show what we can do anyway ----
    if np.all(dphi_L == 0.0) and np.all(dphi_R == 0.0):
        print("=" * 60)
        print("WARNING: This CSV lacks encoder deltas (old format).")
        print("Re-run test_trajectory.py with the updated code that saves")
        print("delta_phi_left and delta_phi_right to get useful model fits.")
        print("=" * 60)
        print()
        print("For now, here is what the data tells us using odometry differencing:")
        print("(less reliable due to odometry already encoding the kinematic model)")

        # Fallback: differencing the odometry to estimate actual v, omega
        dt = np.diff(t)
        dx = np.diff(x_odo)
        dy = np.diff(y_odo)
        dth = np.diff(th_odo)

        v_from_odo = np.sqrt(dx**2 + dy**2) / dt
        w_from_odo = dth / dt

        # Align: v_from_odo[i] results from cmd[i] (approximate, not lag-corrected)
        v_cmd_avg = (vl_cmd[:-1] + vr_cmd[:-1]) / 2
        w_cmd_val = (vr_cmd[:-1] - vl_cmd[:-1]) / BASELINE

        # Skip first few where odometry is zero
        start = 20
        ratio_v = v_from_odo[start:] / v_cmd_avg[start:]
        ratio_w = w_from_odo[start:] / w_cmd_val[start:]
        ratio_w = ratio_w[np.abs(w_cmd_val[start:]) > 0.01]

        print(f"  v_actual / v_cmd  (odo diff): mean={np.mean(ratio_v):.3f}, std={np.std(ratio_v):.3f}")
        if len(ratio_w) > 0:
            print(f"  ω_actual / ω_cmd  (odo diff): mean={np.mean(ratio_w):.3f}, std={np.std(ratio_w):.3f}")

        # Choose best simple gain
        k_v = np.mean(ratio_v)
        k_w = np.mean(ratio_w) if len(ratio_w) > 0 else 1.0

        print(f"\n  Estimated gains: k_v = {k_v:.3f},  k_ω = {k_w:.3f}")
        print(f"  So: v_actual ≈ {k_v:.3f} * (vl+vr)/2")
        print(f"      ω_actual ≈ {k_w:.3f} * (vr-vl)/{BASELINE}")

    # ---- Step 4: Recommend next steps ----
    print("\n" + "=" * 60)
    print("NEXT STEPS (for a reliable trajectory formula):")
    print("=" * 60)
    print("1. Re-run test_trajectory.py with the updated version that saves")
    print("   delta_phi_left and delta_phi_right in the CSV.")
    print()
    print("2. Then re-run this script on the new CSV. It will fit:")
    print("   - Linear:      v_act = k * v_cmd")
    print("   - Affine:      v_act = k * v_cmd + b  (captures friction offset)")
    print("   - Dead-zone:   v_act = k * (v_cmd ± d) (captures stiction)")
    print()
    print("3. The trajectory formula then becomes:")
    print("   v = model_v(vl_cmd, vr_cmd)")
    print("   ω = model_w(vl_cmd, vr_cmd)")
    print("   θ(t+dt) = θ(t) + ω·dt")
    print("   x(t+dt) = x(t) + v·cos(θ + ω·dt/2)·dt")
    print("   y(t+dt) = y(t) + v·sin(θ + ω·dt/2)·dt")
    print()
    print("4. For even better accuracy, save ground-truth poses from")
    print("   AprilTag localization and fit against those instead of odometry.")


if __name__ == "__main__":
    main()
