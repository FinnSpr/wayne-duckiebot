#!/usr/bin/env python3
"""Back out the effective encoder scale factors per step."""
import csv, math
import numpy as np

R = 0.0318
BASELINE = 0.11

with open("values0.csv") as f:
    rows = list(csv.reader(f))
data = [(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in rows]

print("=== Per-step encoder scale factors ===")
print("k = (encoder_delta_phi) / (expected_delta_phi from velocity)")
print("(skipping first 15 steps due to encoder startup delay)\n")

ratios_L = []
ratios_R = []
for i in range(15, len(data) - 1):
    t_curr, vl, vr, x_curr, y_curr, th_curr = data[i]
    t_next = data[i+1][0]
    _, _, _, x_next, y_next, th_next = data[i+1]
    dt = t_next - t_curr
    if dt <= 0:
        continue

    # Back out encoder delta_phi from odometry deltas
    dx = x_next - x_curr
    dy = y_next - y_curr
    dth = th_next - th_curr

    dist_center = math.sqrt(dx**2 + dy**2)
    # dist_right - dist_left = dth * baseline
    # dist_right + dist_left = 2 * dist_center
    # dist_left = dist_center - dth * baseline / 2
    # dist_right = dist_center + dth * baseline / 2
    dist_left  = dist_center - dth * BASELINE / 2
    dist_right = dist_center + dth * BASELINE / 2

    phi_left_enc  = dist_left / R
    phi_right_enc = dist_right / R

    # Expected from commanded velocity (same dt)
    phi_left_exp  = vl * dt / R
    phi_right_exp = vr * dt / R

    if abs(phi_left_exp) > 0.001 and abs(phi_right_exp) > 0.001:
        kL = phi_left_enc / phi_left_exp
        kR = phi_right_enc / phi_right_exp
        ratios_L.append(kL)
        ratios_R.append(kR)

if ratios_L:
    ratios_L = np.array(ratios_L)
    ratios_R = np.array(ratios_R)
    print(f"Left  encoder scale: mean={np.mean(ratios_L):.3f}  std={np.std(ratios_L):.3f}  min={np.min(ratios_L):.3f}  max={np.max(ratios_L):.3f}")
    print(f"Right encoder scale: mean={np.mean(ratios_R):.3f}  std={np.std(ratios_R):.3f}  min={np.min(ratios_R):.3f}  max={np.max(ratios_R):.3f}")
    print(f"Average scale: {np.mean(np.concatenate([ratios_L, ratios_R])):.3f}")

    # If k is constant, the ratio k_L/k_R should be 1
    ratio_LR = ratios_L / ratios_R
    print(f"\nRatio k_L / k_R: mean={np.mean(ratio_LR):.3f}  std={np.std(ratio_LR):.3f}")
    print(f"(Should be ~1.0 if both encoders have same resolution)")

    # Check if the sign is consistent
    print(f"\nLeft  positive fraction: {np.mean(ratios_L > 0):.1%}")
    print(f"Right positive fraction: {np.mean(ratios_R > 0):.1%}")
