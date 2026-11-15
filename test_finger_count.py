"""Run with: python test_finger_count.py"""

import math
from types import SimpleNamespace

from finger_count import count_hand_fingers

SIZE = 640  # square frame so rotating normalised coords is a true rotation

KNUCKLES = {5: (0.40, 0.60), 9: (0.50, 0.58), 13: (0.58, 0.60), 17: (0.65, 0.65)}
TIPS = {8: 5, 12: 9, 16: 13, 20: 17}


def make_hand(open_fingers=(8, 12, 16, 20), thumb_open=True):
    """Upright right-hand-ish pose: wrist at the bottom, fingers pointing up the image."""
    pts = {i: (0.5, 0.9) for i in range(21)}
    pts.update(KNUCKLES)
    for tip, mcp in TIPS.items():
        x, y = KNUCKLES[mcp]
        pts[tip] = (x, y - 0.3) if tip in open_fingers else (x, y + 0.15)
    pts[3] = (0.30, 0.65)
    pts[4] = (0.22, 0.60) if thumb_open else (0.38, 0.68)
    return pts


def transform(pts, degrees=0.0, mirror=False):
    t = math.radians(degrees)
    out = []
    for i in range(21):
        x, y = pts[i][0] - 0.5, pts[i][1] - 0.5
        if mirror:
            x = -x
        out.append(SimpleNamespace(x=0.5 + x * math.cos(t) - y * math.sin(t), y=0.5 + x * math.sin(t) + y * math.cos(t)))
    return out


def test_counts_are_rotation_and_mirror_invariant():
    poses = {
        5: make_hand(),
        0: make_hand(open_fingers=(), thumb_open=False),
        2: make_hand(open_fingers=(8, 12), thumb_open=False),  # peace sign
        1: make_hand(open_fingers=(), thumb_open=True),  # thumbs up
    }
    for expected, pose in poses.items():
        for mirror in (False, True):
            for degrees in range(0, 360, 30):
                got = count_hand_fingers(transform(pose, degrees, mirror), SIZE, SIZE)
                assert got == expected, f"expected {expected}, got {got} at {degrees} deg, mirror={mirror}"


if __name__ == "__main__":
    test_counts_are_rotation_and_mirror_invariant()
    print("ok  test_counts_are_rotation_and_mirror_invariant")
