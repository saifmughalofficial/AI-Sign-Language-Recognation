"""Record your OWN gestures with the webcam (still poses or motions).

    python training/record_custom_gesture.py --label WATER               # still pose
    python training/record_custom_gesture.py --label J --dynamic         # motion (sequence)

Controls (click the video window first):
    SPACE  record one sample (still: instant, motion: records ~1.5 seconds)
    Q      quit

Aim for 40+ samples per gesture, varying hand position and distance a little.
Also record a motion called NONE (hand moving randomly) so the motion model can say "nothing".
Samples are saved as small .npy files (no pictures, no video!) in
    dataset/custom/static/<LABEL>/   or   dataset/custom/dynamic/<LABEL>/
Train them with:   python training/train_model.py --category custom   (or dynamic)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paths import CUSTOM_DIR  # noqa: E402
from vision.camera import Camera  # noqa: E402
from vision.feature_extractor import extract_features, sequence_features  # noqa: E402
from vision.hand_detector import HandDetector, draw_hands  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="name of the gesture, e.g. WATER")
    ap.add_argument("--dynamic", action="store_true", help="record a motion (sequence of frames)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--two-hand", action="store_true", help="still gesture uses two hands")
    args = ap.parse_args()

    import cv2
    label = args.label.strip().upper()
    out_dir = CUSTOM_DIR / ("dynamic" if args.dynamic else "static") / label
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = Camera(args.camera, 640, 480)
    ok, msg = camera.open()
    if not ok:
        print(msg)
        return 1
    detector = HandDetector(2, 0.6, 0.5)
    if not detector.available:
        print(detector.error)
        camera.close()
        return 1

    count = len(list(out_dir.glob("*.npy")))
    recording_until = 0.0
    frames: list[np.ndarray] = []
    handedness = "Right"
    last_id = -1
    print(f"Recording '{label}'. SPACE = record, Q = quit. Existing samples: {count}")
    try:
        while True:
            frame, last_id = camera.read(last_id)
            if frame is None:
                time.sleep(0.005)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            hands = detector.detect(frame)
            shown = cv2.flip(frame, 1)
            draw_hands(shown, hands, mirrored=True)
            if recording_until and hands:
                frames.append(hands[0].landmarks.copy())
                handedness = hands[0].handedness
            if recording_until and time.time() >= recording_until:
                recording_until = 0.0
                feats = sequence_features(frames, handedness)
                if feats is None:
                    print("Too short / no hand - try again.")
                else:
                    np.save(out_dir / f"{int(time.time() * 1000)}.npy", feats)
                    count += 1
                    print(f"Saved motion sample #{count}")
                frames = []
            cv2.putText(shown, f"{label}: {count} samples" + ("  REC" if recording_until else ""),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Record custom gesture (SPACE=record, Q=quit)", shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                if args.dynamic:
                    frames, recording_until = [], time.time() + 1.5
                elif hands:
                    feats = extract_features(hands, two_hand=args.two_hand)
                    np.save(out_dir / f"{int(time.time() * 1000)}.npy", feats)
                    count += 1
                    print(f"Saved sample #{count}")
                else:
                    print("No hand detected - sample not saved.")
    finally:
        camera.close()
        detector.close()
        cv2.destroyAllWindows()
    print(f"Done. {count} samples in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
