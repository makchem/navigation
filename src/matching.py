"""Frame-to-map matching and pose recovery.

Reminder from the roadmap: prefer cv2.estimateAffinePartial2D over
findHomography for nadir imagery. A downward-looking camera over flat terrain
is a 4-DOF similarity (rotation, scale, translation); fitting 8 DOF spends the
extra freedom on noise and can produce confident nonsense on thin inlier sets.
"""


def match_frame(frame, map_img, detector=None):
    """Find correspondences between a drone frame and the reference map.

    Return whatever downstream needs — matched point pairs plus enough
    diagnostics (match counts, distances) to judge quality later.
    """
    raise NotImplementedError


def estimate_pose(frame, map_img, detector=None):
    """Estimate map position (and heading/scale) of `frame`.

    Return a confidence alongside the pose, and make "no fix" representable:
    a wrong position reported confidently is worse than no answer.
    """
    raise NotImplementedError
