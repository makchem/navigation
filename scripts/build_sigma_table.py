"""Generate a sigma lookup table from calibration data and write it to config.

Renders random frames, matches them, buckets by inlier ratio, computes p68
error per bucket, and writes the resulting sigma_lut into config/default.yaml.
"""

import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import yaml

from navmod.augment import augment_frame
from navmod.camera import render_frame
from navmod.georef import distance_m, px_to_latlon
from navmod.matcher import estimate_pose

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRAME_SIZE = (512, 512)
N_SAMPLES = 200
MARGIN = 600
SEED = 42

BUCKET_EDGES = [0.15, 0.25, 0.35, 0.50]


def load_pipeline(config_path="config/default.yaml"):
    with open(os.path.join(REPO_ROOT, config_path)) as f:
        cfg = yaml.safe_load(f)

    map_path = os.path.join(REPO_ROOT, cfg["map"]["path"])
    map_rgba = cv2.imread(map_path, cv2.IMREAD_UNCHANGED)
    if map_rgba is None:
        raise FileNotFoundError(f"Map not found: {map_path}")

    from navmod.georef import MapGeoref

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

    return cfg, map_rgba, map_bgr, georef


def bucket_index(inlier_ratio):
    for i, edge in enumerate(BUCKET_EDGES):
        if inlier_ratio < edge:
            return i
    return len(BUCKET_EDGES)


def main():
    config_path = "config/default.yaml"
    print("Loading map...")
    cfg, map_rgba, map_bgr, georef = load_pipeline(config_path)
    map_h, map_w = map_rgba.shape[:2]
    print(f"Map: {map_w}x{map_h}, GSD={georef.gsd_x:.2f} m/px")

    rng = np.random.RandomState(SEED)

    x_lo, x_hi = MARGIN, map_w - MARGIN
    y_lo, y_hi = MARGIN, map_h - MARGIN

    if x_lo >= x_hi or y_lo >= y_hi:
        print(f"ERROR: map too small for margin={MARGIN}")
        sys.exit(1)

    buckets: dict[int, list[float]] = defaultdict(list)

    print(f"Running {N_SAMPLES} samples (sigma_lut=None, raw performance)...")
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
            bi = bucket_index(fix.inlier_ratio)
            buckets[bi].append(err)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{N_SAMPLES} done")

    n_buckets = len(BUCKET_EDGES) + 1
    edges_with_bounds = [0.0] + BUCKET_EDGES + [1.0]
    sigma_values: list[float | None] = [None] * n_buckets

    for bi in range(n_buckets - 1, -1, -1):
        samples = buckets.get(bi, [])
        if len(samples) >= 5:
            sigma_values[bi] = float(np.percentile(samples, 68))
        elif bi < n_buckets - 1 and sigma_values[bi + 1] is not None:
            sigma_values[bi] = sigma_values[bi + 1] * 1.5
        else:
            sigma_values[bi] = None

    thresholds = []
    lowest_nonempty = None
    for bi in range(n_buckets):
        if buckets.get(bi):
            lowest_nonempty = bi
            break

    for bi in range(n_buckets):
        max_ratio = edges_with_bounds[bi + 1]
        entry: dict = {"max_ratio": round(max_ratio, 2)}

        mark_bad = False
        if lowest_nonempty is not None and bi <= lowest_nonempty:
            if sigma_values[bi] is not None and sigma_values[bi] > 2.0:
                mark_bad = True
            elif sigma_values[bi] is None:
                mark_bad = True

        if mark_bad:
            entry["ok"] = False
        elif sigma_values[bi] is not None:
            entry["sigma_m"] = round(sigma_values[bi], 2)
        else:
            entry["ok"] = False

        thresholds.append(entry)

    cfg["sigma_lut"] = {"thresholds": thresholds}

    config_full_path = os.path.join(REPO_ROOT, config_path)
    with open(config_full_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print("\n--- Sigma Lookup Table ---")
    print(f"{'Bucket':<20} {'Samples':>8} {'p68 (m)':>10} {'LUT sigma':>10} {'ok':>5}")
    print("-" * 58)
    for bi in range(n_buckets):
        lo = edges_with_bounds[bi]
        hi = edges_with_bounds[bi + 1]
        label = f"[{lo:.2f}, {hi:.2f})"
        n = len(buckets.get(bi, []))
        p68 = f"{np.percentile(buckets[bi], 68):.2f}" if n >= 5 else "n/a"
        entry = thresholds[bi]
        ok = entry.get("ok", True)
        sigma_str = f"{entry['sigma_m']:.2f}" if "sigma_m" in entry else "---"
        print(f"{label:<20} {n:>8} {p68:>10} {sigma_str:>10} {str(ok):>5}")

    print(f"\nWritten to {config_full_path}")


if __name__ == "__main__":
    main()
