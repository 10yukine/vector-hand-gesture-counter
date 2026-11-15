"""Orientation-invariant finger counting from MediaPipe's 21 hand landmarks.

Pure Python: works on anything with .x/.y attributes, so it is testable without MediaPipe.
"""

WRIST, THUMB_IP, THUMB_TIP = 0, 3, 4
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
FINGERS = [(8, 5), (12, 9), (16, 13), (20, 17)]  # (tip, MCP knuckle) for index..pinky


def count_hand_fingers(landmarks, width: int, height: int) -> int:
    """Count extended fingers (0-5) on one hand, at any in-plane rotation.

    Landmarks are normalised to [0, 1] per axis, so vectors are scaled back to pixels
    first; otherwise angles are distorted on non-square frames.
    """

    def vec(a: int, b: int):
        return (landmarks[b].x - landmarks[a].x) * width, (landmarks[b].y - landmarks[a].y) * height

    def dot(u, v) -> float:
        return u[0] * v[0] + u[1] * v[1]

    # A finger is open when it points the same way as the palm (wrist -> middle knuckle).
    palm = vec(WRIST, MIDDLE_MCP)
    fingers = sum(dot(palm, vec(mcp, tip)) > 0 for tip, mcp in FINGERS)

    # The thumb moves sideways, so test it against the across-palm axis instead.
    # Pinky knuckle -> index knuckle points toward the thumb side on either hand,
    # which makes this work for left/right hands and palm/back views without a handedness flag.
    # ponytail: a thumb held parallel to the index finger sits near dot == 0 and can flicker;
    # add a margin relative to palm width if that matters.
    thumb = dot(vec(PINKY_MCP, INDEX_MCP), vec(THUMB_IP, THUMB_TIP)) > 0

    return fingers + thumb
