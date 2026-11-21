"""Generate calibration data: render random frames, match, record errors.

Writes data/calibration_results.csv with per-sample ground truth vs estimates,
then prints a summary table.  Used by notebooks/02_sigma_calibration.ipynb to
derive inlier-ratio-based sigma lookup tables.
"""

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np

from navmod.augment import augment_frame
from navmod.camera import render_frame
from navmod.georef import MapGeoref, distance_m, px_to_latlon
from navmod.matcher import estimate_pose

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRAME_SIZE = (512, 512)
N_SAMPLES = 200
MARGIN = 600
SEED = 42


def load_pipeline(config_path="config/default.yaml"):
    import yaml

    with open(os.path.join(REPO_ROOT, config_path)) as f:
        cfg = yaml.safe_load(f)

    map_path = os.path.join(REPO_ROOT, cfg["map"]["path"])
    map_rgba = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
    if map_rgba is None:
        raise FileNotFoundError(f"Map not found: {map_path}")

    h, w = map_rgba.shape[:2]
    georef = MapGeoref(
        top_left_lat=cfg["map"]["top_left_lat"],
        top_left_lon=cfg["map"]["top_left_lon"],
        bottom_right_lat=cfg["map"]["bottom_right_lat"],
        bottom_right_lon=cfg["map"]["bottom_right_lon"],
        image_width_px=w,
        image_height_px=h,
    )

    if map_rgba.shape[2] == 4:
        map_bgr = cv2.cvtColor(map_rgba, cv2.COLOR_BGRA2BGR)
    else:
        map_bgr = map_rgba

    return map_rgba, map_bgr, georef


def main():
    parser = argparse.ArgumentParser(description="Generate calibration data for sigma lookup tables.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    print("Loading map...")
    map_rgba, map_bgr, georef = load_pipeline(args.config)
    map_h, map_w = map_rgba.shape[:2]
    print(f"Map: {map_w}x{map_h}, GSD={georef.gsd_x:.2f} m/px")

    rng = np.random.RandomState(SEED)

    x_lo, x_hi = MARGIN, map_w - MARGIN
    y_lo, y_hi = MARGIN, map_h - MARGIN

    if x_lo >= x_hi or y_lo >= y_hi:
        print(f"ERROR: map too small for margin={MARGIN}")
        sys.exit(1)

    out_path = os.path.join(REPO_ROOT, "data", "calibration_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "true_x", "true_y", "true_lat", "true_lon", "true_heading_deg",
        "est_lat", "est_lon", "est_heading_rad",
        "sigma_m", "ok", "n_inliers", "inlier_ratio", "error_m",
    ]

    errors = []
    ok_count = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(N_SAMPLES):
            x = float(rng.randint(x_lo, x_hi))
            y = float(rng.randint(y_lo, y_hi))
            heading_deg = float(rng.uniform(0, 360))

            true_lat, true_lon = px_to_latlon(georef, x, y)

            frame = render_frame(
                map_rgba, x, y, heading_deg, scale=1.0,
                size=FRAME_SIZE, augment=False,
            )
            if frame.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = frame
            frame_bgr = augment_frame(frame_bgr)

            fix = estimate_pose(frame_bgr, map_bgr, georef, sigma_lut=None)

            if fix.ok:
                err = distance_m(true_lat, true_lon, fix.lat, fix.lon)
                ok_count += 1
                errors.append(err)
                row = dict(
                    true_x=x, true_y=y,
                    true_lat=true_lat, true_lon=true_lon,
                    true_heading_deg=heading_deg,
                    est_lat=fix.lat, est_lon=fix.lon,
                    est_heading_rad=fix.heading_rad,
                    sigma_m=fix.sigma_m, ok=True,
                    n_inliers=fix.n_inliers,
                    inlier_ratio=round(fix.inlier_ratio, 4),
                    error_m=round(err, 3),
                )
            else:
                row = dict(
                    true_x=x, true_y=y,
                    true_lat=true_lat, true_lon=true_lon,
                    true_heading_deg=heading_deg,
                    est_lat=float("nan"), est_lon=float("nan"),
                    est_heading_rad=float("nan"),
                    sigma_m=float("nan"), ok=False,
                    n_inliers=float("nan"),
                    inlier_ratio=float("nan"),
                    error_m=float("nan"),
                )

            writer.writerow(row)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{N_SAMPLES} done")

    no_fix = N_SAMPLES - ok_count
    print(f"\nSaved: {out_path}")
    print(f"Total: {N_SAMPLES}   OK: {ok_count}   No-fix: {no_fix}")

    if errors:
        errors_arr = np.array(errors)
        median = float(np.median(errors_arr))
        p95 = float(np.percentile(errors_arr, 95))
        print(f"Median error: {median:.2f} m")
        print(f"95th %ile error: {p95:.2f} m")
    else:
        print("No successful fixes — cannot compute error stats.")

    print("\nRun scripts/build_sigma_table.py to generate a lookup table from these results.")


if __name__ == "__main__":
    main()
