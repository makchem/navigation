"""Verify the OpenCV environment is usable for map-matching navigation.

Run: uv run scripts/check_env.py
"""

import sys

import cv2
import numpy as np


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


def check_versions() -> None:
    banner("versions")
    print(f"python  {sys.version.split()[0]}")
    print(f"opencv  {cv2.__version__}")
    print(f"numpy   {np.__version__}")
    print(f"threads {cv2.getNumThreads()} / optimized={cv2.useOptimized()}")


def check_detectors() -> None:
    """Feature detectors are the backbone of GPS-free localization:
    match what the camera sees against a reference map tile."""
    banner("feature detectors")
    img = np.zeros((240, 320), np.uint8)
    # A few corners + texture so detectors have something to find.
    cv2.rectangle(img, (40, 40), (160, 150), 255, -1)
    cv2.circle(img, (230, 170), 45, 180, -1)
    cv2.rectangle(img, (190, 30), (260, 90), 120, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # NOTE (OpenCV 5): AKAZE/KAZE/BRISK moved from the main features2d module
    # into xfeatures2d (contrib). cv2.AKAZE no longer exists as it did in 4.x.
    for name, factory in [
        ("ORB", cv2.ORB.create),
        ("SIFT", cv2.SIFT.create),
        ("AKAZE", cv2.xfeatures2d.AKAZE.create),
        ("BRISK", cv2.xfeatures2d.BRISK.create),
    ]:
        try:
            kps, desc = factory().detectAndCompute(img, None)
            shape = desc.shape if desc is not None else None
            print(f"{name:6} ok  keypoints={len(kps):4d} desc={shape}")
        except Exception as exc:  # noqa: BLE001 - report, don't crash the check
            print(f"{name:6} FAIL {type(exc).__name__}: {exc}")


def check_homography() -> None:
    """Estimate the warp between a 'map' and a synthetically shifted 'camera
    view'. This is the exact primitive that turns a match into a position fix."""
    banner("match + homography")
    rng = np.random.default_rng(0)
    map_img = rng.integers(0, 255, (400, 400), dtype=np.uint8)
    map_img = cv2.GaussianBlur(map_img, (5, 5), 0)

    # Known ground truth: rotate 12 deg about centre, then translate.
    truth = cv2.getRotationMatrix2D((200.0, 200.0), 12.0, 1.0)
    truth[:, 2] += (18.0, -9.0)
    view = cv2.warpAffine(map_img, truth, (400, 400))

    sift = cv2.SIFT.create()
    kp1, d1 = sift.detectAndCompute(map_img, None)
    kp2, d2 = sift.detectAndCompute(view, None)

    matcher = cv2.BFMatcher()
    good = [m for m, n in matcher.knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    print(f"keypoints {len(kp1)} vs {len(kp2)}, good matches {len(good)}")

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inliers = int(mask.sum())

    # Compare against ground truth at the image centre.
    probe = np.float32([[[200.0, 200.0]]])
    got = cv2.perspectiveTransform(probe, H)[0, 0]
    want = (truth @ np.array([200.0, 200.0, 1.0]))[:2]
    err = float(np.linalg.norm(got - want))
    print(f"inliers {inliers}/{len(good)}  centre error {err:.2f} px")
    print("homography ok" if err < 2.0 else "homography SUSPECT (error too large)")


def check_optical_flow() -> None:
    """Dead-reckoning between map fixes relies on frame-to-frame flow."""
    banner("optical flow")
    rng = np.random.default_rng(1)
    a = cv2.GaussianBlur(rng.integers(0, 255, (200, 200), dtype=np.uint8), (5, 5), 0)
    b = np.roll(a, (0, 4), axis=(0, 1))  # shift 4 px in x

    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 25, 3, 5, 1.2, 0)
    dx = float(np.median(flow[20:-20, 20:-20, 0]))
    print(f"Farneback median dx = {dx:+.2f} px (expected ~+4.00)")

    pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01, minDistance=5)
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
    tracked = int(status.sum())
    print(f"Lucas-Kanade tracked {tracked}/{len(pts)} points")


def check_io() -> None:
    banner("io + gui")
    img = np.full((32, 32, 3), (40, 90, 200), np.uint8)
    ok, buf = cv2.imencode(".png", img)
    print(f"imencode/imdecode {'ok' if ok and cv2.imdecode(buf, 1).shape == img.shape else 'FAIL'}")

    build = cv2.getBuildInformation()
    for key in ("GUI:", "FFMPEG:", "V4L/V4L2"):
        line = next((l.strip() for l in build.splitlines() if key in l), None)
        print(f"{key:10} {line or 'not reported'}")

    cams = [i for i in range(4) if _camera_works(i)]
    print(f"cameras    working indices: {cams or 'none found'}")
    print("note: imshow needs a GUI session; run scripts/camera_test.py to see a live window")


def _camera_works(index: int) -> bool:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    try:
        return bool(cap.isOpened() and cap.read()[0])
    finally:
        cap.release()


if __name__ == "__main__":
    check_versions()
    check_detectors()
    check_homography()
    check_optical_flow()
    check_io()
    print("\nenvironment check complete")
