"""Validate map imagery in data/ before a notebook wastes time on it.

cv2.imread returns None on a corrupt or missing file instead of raising, so a
bad download surfaces later as a confusing "'NoneType' has no attribute shape"
several lines away from the real cause. This checks structure *and* a full
decode, and says which one failed.

Run: uv run scripts/check_data.py [path ...]
"""

import sys
from pathlib import Path

import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # a 5 km map legitimately exceeds the bomb limit

DATA = Path(__file__).resolve().parent.parent / "data"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def check_structure(path: Path) -> list[str]:
    """Cheap byte-level checks that catch an interrupted download."""
    problems = []
    data = path.read_bytes()

    if data.startswith(PNG_MAGIC):
        # A complete PNG ends with the 12-byte IEND chunk.
        if b"IEND" not in data[-16:]:
            tail = data[-4:]
            hint = f" (ends inside a {tail.decode('ascii', 'replace')} chunk)" if tail.isalpha() else ""
            problems.append(f"PNG has no IEND chunk — file is truncated{hint}")
    elif data[:2] == b"\xff\xd8":
        # JPEG must end with the EOI marker.
        if data[-2:] != b"\xff\xd9":
            problems.append("JPEG has no EOI marker — file is truncated")
    else:
        problems.append(f"unrecognized format, first bytes {data[:8].hex()}")

    return problems


def check_decode(path: Path) -> list[str]:
    """Force a full pixel decode — the only way to be sure it is usable."""
    problems = []

    try:
        with Image.open(path) as im:
            im.load()
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        problems.append(f"PIL decode failed: {type(exc).__name__}: {exc}")

    if cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
        problems.append("cv2.imread returned None")

    return problems


def report(path: Path) -> bool:
    print(f"\n{path.name}")
    if not path.exists():
        print("  MISSING")
        return False

    size_mb = path.stat().st_size / 1e6
    print(f"  {size_mb:.1f} MB")

    problems = check_structure(path) + check_decode(path)
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        return False

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    ch = 1 if img.ndim == 2 else img.shape[2]
    print(f"  OK    {w} x {h} px ({w * h / 1e6:.1f} MPix), {ch} channels, {img.dtype}")
    for km in (5, 10):
        print(f"        if {km} km wide -> GSD {km * 1000 / w:.3f} m/px")
    return True


if __name__ == "__main__":
    args = sys.argv[1:]
    paths = (
        [Path(a) for a in args]
        if args
        else sorted(p for p in DATA.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    )

    if not paths:
        print(f"no imagery found in {DATA}")
        raise SystemExit(1)

    ok = all(report(p) for p in paths)
    print("\nall files usable" if ok else "\nfix the files marked FAIL before running the notebooks")
    raise SystemExit(0 if ok else 1)
