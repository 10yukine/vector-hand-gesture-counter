"""Multi-hand finger counter: MediaPipe hand landmarks + vector dot-product finger test.

Run: python hand_tracker.py   ([D] toggle skeleton, [Q] quit)
"""

import argparse
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from finger_count import WRIST, count_hand_fingers

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]  # fmt: skip
WINDOW = "Vector Hand Tracker"


def ensure_model(model_path: str) -> str:
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        print(f"Downloading {model_path} from the MediaPipe model repository...")
        tmp_path = model_path + ".part"  # so an interrupted download never looks like a valid model
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        os.replace(tmp_path, model_path)
    return model_path


def draw_skeleton(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (255, 0, 0), cv2.FILLED)


def main(camera_idx: int, model_path: str, max_hands: int) -> None:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=ensure_model(model_path)),
        num_hands=max_hands,
        running_mode=vision.RunningMode.VIDEO,
    )
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_idx}. Try a different --camera index.")

    show_skeleton = True
    timestamp_ms = 0
    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # VIDEO mode rejects timestamps that do not strictly increase.
            timestamp_ms = max(timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            total = 0
            for landmarks in result.hand_landmarks:
                count = count_hand_fingers(landmarks, w, h)
                total += count
                if show_skeleton:
                    draw_skeleton(frame, landmarks)
                wrist = (int(landmarks[WRIST].x * w), int(landmarks[WRIST].y * h))
                cv2.putText(frame, str(count), wrist, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            cv2.putText(
                frame,
                f"Fingers: {total} | Hands: {len(result.hand_landmarks)} | [D] Skeleton: {'ON' if show_skeleton else 'OFF'}",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector Dot-Product Hand & Finger Counter")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--model", default="hand_landmarker.task", help="Model path (downloaded if missing)")
    parser.add_argument("--max-hands", type=int, default=4)
    args = parser.parse_args()
    main(args.camera, args.model, args.max_hands)
