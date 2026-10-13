"""Dataset & model setup helper. Run:   python setup_dataset.py

With no options it explains, step by step:
  1. where to download a FREE dataset      2. where to put it
  3. how to prepare it                     4. how to train
  5. where the trained model is saved
and shows what is already installed on this computer.

Options:
  --check                  only show the status table
  --generate-images        create the schematic Help pictures in assets/gestures
  --download-vosk          download the small offline English speech model (~40 MB, internet once)
  --download-hand-model    download hand_landmarker.task (only needed for newer MediaPipe versions)
  --run-pipeline           run prepare -> extract -> train in one go
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.constants import MODEL_FILES  # noqa: E402
from core.paths import (CUSTOM_DIR, GESTURE_IMG_DIR, HAND_TASK_FILE, MODELS_DIR, PROCESSED_DIR,  # noqa: E402
                        RAW_DIR, ROOT, ensure_dirs)

VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/1/hand_landmarker.task")

GUIDE = f"""
==============================================================================
 SIGN LANGUAGE ASSISTANT - DATASET & MODEL SETUP  (everything here is free)
==============================================================================
The app works right now in DEMO MODE (no dataset needed). For real recognition
you train a small model on a public dataset. This happens on your own computer.

STEP 1 - DOWNLOAD A FREE DATASET  (do this by hand in your browser; a free
         account on the site may be required. Read each dataset's licence.)

   A) ASL Alphabet (letters A-Z)        Kaggle  -  search: "ASL Alphabet" by grassknoted
        https://www.kaggle.com/datasets/grassknoted/asl-alphabet
   B) Roboflow Universe public ASL sets (from your blueprint file)
        https://universe.roboflow.com/dk-guoqz/asl-dataset-p9yw8-ay3wr
        https://universe.roboflow.com/sign-language-yhtzn/sign-language-mc7fl
        Export format: "Folder Structure" (classification) or "YOLOv8" - both work.
   C) Numbers 0-9                       Kaggle  -  search: "Sign Language Digits Dataset"
   D) Hugging Face Datasets             search "asl alphabet" at https://huggingface.co/datasets
   E) Words: there is no good free still-image dataset for words. Record your own
      with:  python training/record_custom_gesture.py --label WATER

   Tip: you do NOT need the whole dataset. 200-300 pictures per sign is plenty.

STEP 2 - PUT IT HERE
        {RAW_DIR}
   Unzip so that pictures sit in folders named after the sign, for example
        dataset/raw/asl_alphabet_train/A/A1.jpg
        dataset/raw/asl_alphabet_train/B/B1.jpg
   (Roboflow YOLO exports with a data.yaml file are recognised automatically.)

STEP 3 - PREPARE
        python training/prepare_dataset.py
        python training/extract_features.py

STEP 4 - TRAIN
        python training/train_model.py           (or double-click train_model.bat)
   It prints the accuracy on pictures the model has never seen.

STEP 5 - WHERE THE MODEL IS SAVED
        {MODELS_DIR}
        asl_alphabet_model.pkl  numbers_model.pkl  words_model.pkl
   Restart the app: the "Demo Mode" banner disappears.

Being honest: a model trained on one dataset will work best for people/backgrounds
that look like that dataset. It is a prototype, not a certified translator.
"""


def check() -> None:
    print("\nSTATUS")
    print("-" * 60)
    raw_count = sum(1 for f in RAW_DIR.rglob("*.*") if not f.name.startswith(".")) if RAW_DIR.exists() else 0
    print(f"  dataset/raw            : {raw_count} files" + ("" if raw_count else "   (empty - see STEP 1-2)"))
    for cat in ("alphabet", "numbers", "words"):
        folder = PROCESSED_DIR / cat
        n = sum(1 for _ in folder.rglob("*.*") if _.suffix.lower() in (".jpg", ".jpeg", ".png")) if folder.exists() else 0
        print(f"  processed/{cat:<9}    : {n} pictures")
    print("  trained models:")
    for cat, name in MODEL_FILES.items():
        print(f"     {'OK ' if (MODELS_DIR / name).exists() else '-- '} {name}")
    demo = not any((MODELS_DIR / MODEL_FILES[c]).exists() for c in ("alphabet", "number", "word", "custom"))
    print(f"  app mode               : {'DEMO MODE (no trained model yet)' if demo else 'trained model(s) found'}")
    print(f"  custom recordings      : {sum(1 for _ in CUSTOM_DIR.rglob('*.npy')) if CUSTOM_DIR.exists() else 0} samples")
    vosk_dirs = [p.name for p in MODELS_DIR.glob("vosk*") if p.is_dir()]
    print(f"  offline speech model   : {vosk_dirs[0] if vosk_dirs else 'not installed (--download-vosk)'}")
    imgs = len(list(GESTURE_IMG_DIR.glob("*.png"))) if GESTURE_IMG_DIR.exists() else 0
    print(f"  help pictures          : {imgs} in assets/gestures")
    try:
        import mediapipe as mp
        legacy = hasattr(mp, "solutions")
        print(f"  MediaPipe              : {mp.__version__} ({'ready' if legacy else 'new version - needs hand_landmarker.task'})")
        if not legacy:
            print(f"     hand model file      : {'OK' if HAND_TASK_FILE.exists() else 'missing (--download-hand-model)'}")
    except Exception:  # noqa: BLE001
        print("  MediaPipe              : not installed (pip install -r requirements.txt)")


def download(url: str, target: Path) -> bool:
    print(f"Downloading {url}\n  -> {target}")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp, open(target, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while chunk := resp.read(1 << 16):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done * 100 // total}%", end="", flush=True)
        print()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\nDownload failed: {exc}\nCheck your internet connection or download it manually from the link above.")
        return False


def download_vosk() -> None:
    zip_path = MODELS_DIR / "vosk-model.zip"
    if download(VOSK_URL, zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(MODELS_DIR)
        zip_path.unlink(missing_ok=True)
        print("Speech model installed. Voice Mode is now available offline.")


def run_pipeline() -> int:
    for script in ("prepare_dataset.py", "extract_features.py", "train_model.py"):
        print(f"\n>>> {script}")
        code = subprocess.call([sys.executable, str(ROOT / "training" / script)])
        if code != 0:
            return code
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--generate-images", action="store_true")
    ap.add_argument("--download-vosk", action="store_true")
    ap.add_argument("--download-hand-model", action="store_true")
    ap.add_argument("--run-pipeline", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    acted = False
    if args.generate_images:
        from core.registry import GestureRegistry
        from ui.gesture_art import generate_all
        print(f"Created {generate_all(GestureRegistry())} help pictures in assets/gestures")
        acted = True
    if args.download_vosk:
        download_vosk()
        acted = True
    if args.download_hand_model:
        download(HAND_TASK_URL, HAND_TASK_FILE)
        acted = True
    if args.run_pipeline:
        return run_pipeline()
    if not acted and not args.check:
        print(GUIDE)
    check()
    return 0


if __name__ == "__main__":
    sys.exit(main())
