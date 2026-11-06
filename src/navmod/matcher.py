"""Frame-to-map matching and pose recovery.

Reminder from the roadmap: prefer cv2.estimateAffinePartial2D over
findHomography for nadir imagery. A downward-looking camera over flat terrain
is a 4-DOF similarity (rotation, scale, translation); fitting 8 DOF spends the
extra freedom on noise and can produce confident nonsense on thin inlier sets.
"""
import cv2, numpy as np

def match_frame(frame, map_img, detector=None):
    """Find correspondences between a drone frame and the reference map.

    Return whatever downstream needs — matched point pairs plus enough
    diagnostics (match counts, distances) to judge quality later.
    """

    orb = cv2.ORB.create(nfeatures=3000)
    kp_map, des_map = orb.detectAndCompute(map_img, None)
    kp_frame, des_frame = orb.detectAndCompute(frame, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des_frame, des_map, k=2)

    good = [m for m, n in matches if m.distance < 0.65 * n.distance]

    pts_frame = np.float32([kp_frame[m.queryIdx].pt for m in good])
    pts_map = np.float32([kp_map[m.trainIdx].pt for m in good])

    if len(good) < 10:
        return None
    
    M, inliers = cv2.estimateAffinePartial2D(pts_frame, pts_map)

    h, w = frame.shape[:2]
    center_frame = np.array([w/2, h/2, 1])
    center_map = M @ center_frame

    return center_map


def estimate_pose(frame, map_img, detector=None):
    """Estimate map position (and heading/scale) of `frame`.

    Return a confidence alongside the pose, and make "no fix" representable:
    a wrong position reported confidently is worse than no answer.
    """
    raise NotImplementedError
