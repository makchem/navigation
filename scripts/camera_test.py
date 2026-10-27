"""Live camera window with ORB keypoints overlaid — confirms capture + GUI work.

Run: uv run scripts/camera_test.py [camera_index]
Keys: q or ESC to quit, SPACE to save a frame into out/
"""

import sys
from pathlib import Path

import cv2

OUT = Path(__file__).resolve().parent.parent / "out"


def main(index: int = 0) -> int:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"could not open camera {index}", file=sys.stderr)
        return 1

    orb = cv2.ORB.create(nfeatures=400)
    saved = 0
    print("q/ESC quit, SPACE save frame")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed", file=sys.stderr)
                return 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kps = orb.detect(gray, None)
            vis = cv2.drawKeypoints(frame, kps, None, color=(0, 255, 0))
            cv2.putText(
                vis, f"cam {index}  {len(kps)} keypoints", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
            cv2.imshow("camera test", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                OUT.mkdir(exist_ok=True)
                path = OUT / f"frame_{saved:03d}.png"
                cv2.imwrite(str(path), frame)
                print(f"saved {path}")
                saved += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
