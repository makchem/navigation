"""Regression tests that the OpenCV install stays usable for map matching.

These are headless-safe (no imshow, no camera) so they run anywhere.
"""

import cv2
import numpy as np
import pytest


def test_imports_and_version():
    assert int(cv2.__version__.split(".")[0]) >= 5


@pytest.mark.parametrize(
    "factory",
    [cv2.ORB.create, cv2.SIFT.create, cv2.xfeatures2d.AKAZE.create, cv2.xfeatures2d.BRISK.create],
)
def test_detector_produces_descriptors(factory):
    img = np.zeros((240, 320), np.uint8)
    cv2.rectangle(img, (40, 40), (160, 150), 255, -1)
    cv2.circle(img, (230, 170), 45, 180, -1)

    kps, desc = factory().detectAndCompute(cv2.GaussianBlur(img, (3, 3), 0), None)
    assert len(kps) > 0
    assert desc is not None and desc.shape[0] == len(kps)


def test_homography_recovers_known_transform():
    rng = np.random.default_rng(0)
    map_img = cv2.GaussianBlur(rng.integers(0, 255, (400, 400), dtype=np.uint8), (5, 5), 0)

    truth = cv2.getRotationMatrix2D((200.0, 200.0), 12.0, 1.0)
    truth[:, 2] += (18.0, -9.0)
    view = cv2.warpAffine(map_img, truth, (400, 400))

    sift = cv2.SIFT.create()
    kp1, d1 = sift.detectAndCompute(map_img, None)
    kp2, d2 = sift.detectAndCompute(view, None)
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    assert len(good) >= 50

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)

    got = cv2.perspectiveTransform(np.float32([[[200.0, 200.0]]]), H)[0, 0]
    want = (truth @ np.array([200.0, 200.0, 1.0]))[:2]
    assert np.linalg.norm(got - want) < 2.0


def test_optical_flow_recovers_known_shift():
    rng = np.random.default_rng(1)
    a = cv2.GaussianBlur(rng.integers(0, 255, (200, 200), dtype=np.uint8), (5, 5), 0)
    b = np.roll(a, 4, axis=1)

    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 25, 3, 5, 1.2, 0)
    assert np.median(flow[20:-20, 20:-20, 0]) == pytest.approx(4.0, abs=0.5)


def test_pyproj_roundtrip():
    """Map pixel work eventually needs real-world coords; make sure the
    projection stack is wired up too."""
    from pyproj import Transformer

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    back = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

    lon, lat = 13.405, 52.52  # Berlin
    x, y = to_utm.transform(lon, lat)
    lon2, lat2 = back.transform(x, y)
    assert (lon2, lat2) == pytest.approx((lon, lat), abs=1e-6)
