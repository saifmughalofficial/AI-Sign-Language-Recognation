# 🤟 Sign Language Assistant (offline ASL communication prototype)

A **free, offline, privacy-first** desktop app that helps a person who uses hand signs
talk with a person who does not know sign language.

* **Person A** signs to the webcam → the app recognises hand shapes → builds a sentence → **speaks it aloud**.
* **Person B** speaks (or types) → the app shows the words as **text on screen**.

> **Please read this honestly:** this is a *mock / learning prototype*. It does **not** understand
> arbitrary ASL. Real sign language uses movement, two hands, facial expression and grammar.
> The project is built so it can grow toward that, and it clearly labels what is real, what is
> experimental and what is not supported (see [Supported gestures](#supported-gestures)).

**Everything is free.** No paid API, no account, no credit card, no cloud. All recognition runs on your laptop CPU.

---

## Contents
1. [Features](#features) · 2. [Screenshots](#screenshots) · 3. [Install (Windows)](#installation) ·
4. [Run + Demo Mode](#running-the-app) · 5. [Dataset setup](#dataset-setup) · 6. [Training](#training) ·
7. [Supported gestures](#supported-gestures) · 8. [How it works](#architecture) ·
9. [Adding gestures / phrases / languages](#adding-new-gestures) · 10. [Running fully offline](#running-completely-offline) ·
11. [Troubleshooting](#troubleshooting) · 12. [Privacy](#privacy) · 13. [Tests](#tests) · 14. [Future ideas](#future-improvements)

---

## Features
* **Works out of the box — no training needed.** A rule-based recogniser measures the 21 MediaPipe landmarks
  (3-D joint angles, fingertip contact, finger spread / crossing, pointing direction) and recognises
  **A–Z** (J and Z by motion), **0–9**, **23 words** (incl. STOP, WAIT, WATER, HELLO, YES, NO, PLEASE, HELP…),
  **two-hand signs** and **10 one-gesture sentences**. Trained models (below) still plug in and take priority.
* Live webcam with a glowing, colour-coded hand skeleton, hold-progress ring, floating "+ WORD" toasts,
  FPS and status lights (Camera / Hand / Recognition).
* Modes: **ALPHABET, NUMBERS, WORDS, SENTENCES, AUTO** (keys `1`–`5`).
* **Gesture stabilisation** – a gesture must be held ~0.7 s and is *never* added 30 times per second;
  a 5-frame vote removes flicker.
* **Animations** – splash + fade-in, pop-in gesture label, gliding meters, typewriter sentence, card flash on accept,
  pulsing LIVE dot, rotating "Try…" tips ribbon, collapsing banner, hover glow on buttons.
* **Sentence builder** – `I + NEED + WATER → "I need water."`, `A + L + I → USE AS NAME → "My name is Ali."`
  SPACE / DELETE / CLEAR / CAPITALIZE / ADD WORD, name box, quick phrases.
* **Offline text-to-speech** (pyttsx3): SPEAK, STOP SPEECH, REPEAT, voice choice, optional letter / word / sentence speech with cooldown.
* **Voice mode** (Vosk, offline) + typed fallback for Person B. Two-way communication panel + local history.
* **Help Center** with a visual, searchable gesture dictionary and a separate **OFFLINE HELP** (plain text).
* **Settings** saved in `config/settings.json` (camera, resolution, confidences, speech, theme, stabilisation time…).
* **Demo Mode** when no trained model exists – the whole app flow works, clearly labelled *"Demo prediction"*.
* **Experimental motion gestures** (J, Z, and your own recorded motions) using sequences of landmarks.
* Dark / light theme, large buttons, high contrast, keyboard shortcuts: `C` camera · `S` speak · `H` help · `G` gesture guide · `V` voice · `R` reset · `1`–`5` mode · `Esc` close.
* Never crashes because a webcam, microphone, model, voice or file is missing – it shows a friendly message.

## Screenshots
Put your own screenshots in `assets/screenshots/` (for example `dashboard.png`) and link them here:

| Dashboard | Help Center | Settings |
|---|---|---|
| *(add dashboard.png)* | *(add help.png)* | *(add settings.png)* |

## Installation
You need **Windows 10/11** and **Python 3.11** ([python.org](https://www.python.org/downloads/) – tick *"Add python.exe to PATH"*).
Python 3.10–3.12 also work.

**Easy way:** double-click `setup.bat`, then `run_app.bat`.

**Manual way** (same thing, step by step):
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python setup_dataset.py --generate-images
python app.py
```
`requirements.txt` only contains free, open-source packages: OpenCV, MediaPipe (pinned to `0.10.14`), NumPy,
scikit-learn, joblib, Pillow, pyttsx3, vosk, sounddevice. Tkinter comes with Python.

*If `vosk` or `sounddevice` fail to install*, delete those two lines – everything except spoken input still works.

### Optional: offline speech recognition model (for Voice Mode)
```bat
python setup_dataset.py --download-vosk
```
(needs internet **once**; ~40 MB). Or download `vosk-model-small-en-us-0.15` yourself from
<https://alphacephei.com/vosk/models> and unzip it into `models/`.

## Running the app
```bat
python app.py          (or double-click run_app.bat)
```
1. A splash screen appears, then the dashboard. The **camera starts OFF** (privacy).
2. Press **START CAMERA** → the preview appears top-left.
3. Show your hand, hold a gesture steady until the bar fills → it is added to the sentence.
4. Press **SPEAK** → the sentence is spoken with an offline voice.

### Smart recognition (no dataset needed)
Without a trained model the app uses **rule-based shape recognition** (`vision/hand_shape.py` + `vision/shape_library.py`).
Every frame, the landmarks are turned into measurements (how bent each finger is, where the thumb is, which
fingertips touch, spread, direction) and each hand shape gets a **score 0–100 %** that is shown as the confidence.
* Press **GESTURE GUIDE** (or `G`) to see every pose for the current mode. The tips ribbon under the mode buttons rotates through them.
* Tips for good detection: good light, hand ~50–80 cm from the camera, palm towards the camera, whole hand in the picture.
* **RUN DEMO** plays a scripted example without a camera (words → sentence gestures → spelling a name).
* Which pose means which gesture is data, not code: change `"shape"` in `data/gestures.json` (shape names are listed in `SHAPE_HINTS`).

## Dataset setup
Run `python setup_dataset.py` for a printed guide plus a status table. Summary:

1. **Download** a free dataset yourself (accounts may be required; check licences – we never scrape or bypass anything):
   * Kaggle **ASL Alphabet** (A–Z) · Roboflow Universe public ASL datasets (links in your blueprint; export *Folder Structure* or *YOLOv8*)
   * Kaggle **Sign Language Digits** (0–9) · Hugging Face datasets (search "asl alphabet")
   * **Words:** no good free still-image dataset exists → record your own (below).
2. **Put it in** `dataset/raw/` (folders named after the sign: `A/`, `B/`, … `0`–`9`).
3. `python training/prepare_dataset.py` → sorts pictures into `dataset/processed/alphabet|numbers|words/`
   (300 per sign by default; J and Z are skipped because they need movement).
4. `python training/extract_features.py` → MediaPipe finds the hand in each picture; numbers are saved to `.npz` files.

Folder layout:
```
dataset/raw/        <- you put the download here
dataset/processed/  alphabet/  numbers/  words/  features_*.npz
dataset/custom/     static/<LABEL>/*.npy   dynamic/<LABEL>/*.npy   (your own recordings)
```

## Training
```bat
train_model.bat                                    (runs prepare → extract → train)
python training/train_model.py --category alphabet --algo svm     (rf | svm | knn)
python training/evaluate_model.py --category alphabet             (accuracy + common mix-ups)
```
Trained models are saved to `models/`: `asl_alphabet_model.pkl`, `numbers_model.pkl`, `words_model.pkl`
(+ `custom_model.pkl`, `dynamic_model.pkl` for your own gestures). Restart the app – Demo Mode disappears.

Each model is a scikit-learn pipeline (`StandardScaler` normalisation + Random Forest / SVM / KNN). Training prints
accuracy on **held-out** pictures. Expect it to work best on people and backgrounds similar to the dataset.
Similar shapes (M/N/S/T, 0/O, F/9, V/2, U/R) are often confused – that is normal for still-picture models.

### Record your own gestures
```bat
python training/record_custom_gesture.py --label WATER               (still pose, press SPACE ~40 times)
python training/record_custom_gesture.py --label J --dynamic         (motion; also record a motion called NONE)
python training/train_model.py --category custom       (or: --category dynamic)
```
Only small `.npy` number files are saved – **no pictures or video**.

## Supported gestures
| Level | What | Notes |
|---|---|---|
| **Camera ✓ (no training)** | Letters **A–Z** (24 still + J/Z motion), numbers **0–9**, 23 words, 10 sentence gestures | Rule-based shapes. M / N / S / T are the hardest (they differ only by where the thumb hides). |
| **Trained model (optional)** | Anything in your dataset | Takes priority over the rules for that category. |
| **Type-only** | HI, HOME, SCHOOL, HOSPITAL, DOCTOR, GO, COME, LEAVE, ME, LIKE, THIRSTY, IS | These ASL signs need movement – add them with *Add word*, or record them yourself. |
| **Unsupported** | Full ASL grammar, facial expressions, fast fingerspelling, other sign languages | Not attempted. |

**Camera words (WORDS mode):** HELLO ✋ open hand · STOP flat palm · WAIT 🙌 both hands open · WATER W hand ·
YES fist · NO index+middle touch thumb · PLEASE 🙏 palms together · THANK YOU flat hand + thumb out · SORRY A-hand ·
HELP 👍 on open palm · GOOD 👍 · BAD 👎 · LOVE 🤟 · NAME H sideways · MY flat hand sideways · I pinky up ·
YOU 👉 · NEED X hook · WANT claw · WHERE ☝ · FOOD flat-O · HUNGRY C hand · CALL 🤙.
Signing `I + NEED + WATER` gives *"I need water."*, `PLEASE + STOP` gives *"Please stop!"*.

**Sentence gestures (SENTENCES mode – one pose = one whole sentence, spoken automatically):**

| Pose | Sentence |
|---|---|
| ✋ open hand, fingers spread | Hello, how are you? |
| W hand (3 fingers) | I need water. |
| flat-O (fingertips on thumb) | I am hungry. I need food. |
| H hand sideways | My name is *{name from the Name box}*. |
| flat palm, fingers together | Please stop! |
| 🙌 both hands open | Please wait a moment. |
| 👍 thumbs up | I am fine, thank you. |
| 🤙 thumb + pinky | Please call a doctor. |
| 🤟 I-love-you hand | I love you. |
| 👍 resting on the other open palm | Please help me. |

> The Help pictures are **schematic illustrations** generated locally (which fingers are up, hand direction, motion, one/two hands),
> not photographs of official signs. To use real photos, save a PNG with the same name in `assets/gestures/`
> (e.g. `word_WATER.png`, `alphabet_A.png`) – it is shown automatically.

## Architecture
```
Webcam → OpenCV → MediaPipe Hands → 21 landmarks → feature extraction → normalisation
       → local classifier (scikit-learn)  ┐
       → (experimental) motion recogniser ┴→ GestureStabilizer → SentenceBuilder + PhraseEngine → Text-to-speech
Microphone → Vosk (offline) → text → Person B panel / history
```
```
app.py                  entry point (splash → dashboard)
core/                   settings, history, registry of JSON data, shared constants, AppContext (wiring)
vision/  camera.py hand_detector.py feature_extractor.py gesture_recognizer.py gesture_stabilizer.py pipeline.py
speech/  text_to_speech.py speech_to_text.py
sentence/ sentence_builder.py phrase_engine.py
ui/      dashboard.py help_window.py settings_window.py gesture_dictionary.py gesture_art.py theme.py
training/ prepare_dataset.py extract_features.py train_model.py evaluate_model.py record_custom_gesture.py
data/    gestures.json phrases.json help.json vocabulary.json     ← all knowledge lives in plain JSON
tests/   unit tests (run offline, no camera needed)
```
Design notes for developers:
* `vision/pipeline.py` runs camera + AI in a **background thread**; the window never freezes. AI runs on every N-th frame (setting) for CPU efficiency.
* Landmarks are wrist-centred and size-normalised, so hand position and distance do not matter. Left hands are mirrored.
  Training pictures and live frames are both given to MediaPipe **un-flipped** so left/right labels match. Only the *display* is mirrored.
* **Static vs dynamic:** static gestures use one frame; dynamic gestures use `sequence_features()` (16 resampled frames) and
  `DynamicGestureRecognizer` (path templates for J/Z or your trained sequence model). Both plug into the same stabiliser and sentence builder.
* MediaPipe compatibility: `mediapipe==0.10.14` (built-in model). Newer versions are supported via the *Tasks* API if you run
  `python setup_dataset.py --download-hand-model`.

## Adding new gestures
Static gesture (e.g. **WATER** as a word) – add one object to `data/gestures.json`:
```json
{
  "gesture": "TOILET", "category": "word", "type": "static", "hands": 1, "status": "experimental",
  "instructions": "Make a T hand and shake it gently.",
  "sentence_examples": ["Where is the toilet?"],
  "demo_pose": null, "image_fingers": "00000", "orientation": "up", "asl_motion": true,
  "image": "word_TOILET.png"
}
```
* `category`: `alphabet | number | word | custom` · `type`: `static | dynamic` · `hands`: `1 | 2`
* `demo_pose` (optional): five 0/1 characters (thumb→pinky) for Demo Mode. Must be unique inside its category.
* `image_fingers`: five characters, `1` up, `0` folded, `2` curved – used to draw the Help picture. Then run
  `python setup_dataset.py --generate-images`.
* Add the word to `data/vocabulary.json` so it appears in the ADD WORD list.
* To *recognise* it for real: record samples with `record_custom_gesture.py` and train `--category custom`
  (motion: add `--dynamic`, train `--category dynamic`). **No Python code changes are needed.**
* Two-hand gestures: set `"hands": 2` and record with `--two-hand`.

## Adding new phrases
Edit `data/phrases.json`:
* `templates` – e.g. `{"match": ["WHERE", "{w}"], "output": "Where is the {w}?"}` (`{n}` = a name, `{w}` = any word, `{name}` = the configured name;
  add `"alone": true` if it should only apply when it is the whole sentence)
* `copula_words` – words that get "am/are/is" (I + HUNGRY → "I am hungry")
* `phrase_library` – the list shown in Help → Phrases

## Adding a new language
Add a block under `languages` in `data/vocabulary.json` (e.g. `"ur"`), add the code to `LANGUAGES` in `core/settings.py`,
and install a matching Vosk model for spoken input. (The interface text itself is English in this prototype.)

## Running completely offline
1. On any internet-connected moment **once**: `pip install -r requirements.txt`, optionally `--download-vosk`, and download a dataset.
2. After that, disconnect. Camera, recognition, training, speech, voice mode, help and history need **no internet**.
3. Only these steps ever touch the internet and both are optional one-time downloads: dataset download (by you, in a browser) and `setup_dataset.py --download-vosk / --download-hand-model`.
4. Text-to-speech uses the voices already installed in Windows (SAPI5). Add more under *Windows Settings → Time & Language → Speech*.

## Troubleshooting
| Problem | Fix |
|---|---|
| Camera will not start | Close Zoom/Teams/Camera app; try another camera number in Settings; check Windows *Privacy → Camera* permission. |
| "No hand detected" | Move your hand into view, add light in front of you, lower *Detection confidence* slightly. |
| "Gesture unclear" | Hold still; move a bit closer; plain background helps. |
| Letters repeat or are missed | Adjust *Gesture stabilization* and *Repeat delay* in Settings. For double letters (LL) lower your hand briefly between them. |
| Demo Mode will not go away | `models/` must contain the exact file names listed above; run `python setup_dataset.py --check`. |
| Model "could not be loaded" | It was made with another scikit-learn/feature version → retrain with `train_model.bat`. |
| No sound | Check Windows volume; pick another voice in Settings → TEST VOICE. |
| Voice Mode unavailable | Install `vosk sounddevice`, download the model (`--download-vosk`), check the microphone. Person B can type meanwhile. |
| `mediapipe` install fails | Use Python 3.10–3.12 64-bit. Newer MediaPipe works too after `--download-hand-model`. |
| `A module was compiled with NumPy 1.x` | `pip install "numpy<2"` |
| Slow / laggy | Resolution 320x240 or 640x480, raise *Process every N frames*, close other apps. |
| Tkinter missing | Re-install Python and keep *tcl/tk and IDLE* ticked. |

## Privacy
> All camera processing and recognition is performed locally. No camera frames, microphone recordings or conversations
> are uploaded to a cloud service.

* Camera is OFF until you press START. No video/audio is stored. Nothing is uploaded.
* History is saved only in `config/history.json` (delete it with **CLEAR HISTORY** or switch off *Save history*).
* Custom gesture recordings are stored as anonymous landmark numbers, never pictures.
* Downloading datasets/models is optional and done by you.

## Tests
```bat
python -m unittest discover -s tests -v          (or: pip install pytest && pytest)
```
Covers: gesture stabilisation & duplicate prevention, sentence building, delete/clear, phrase generation, text-to-speech (fake engine),
settings validation, help dictionary/search, missing/corrupt model handling, a small end-to-end training run on synthetic hands, motion detection.
The tests need no camera, microphone or internet.

## Future improvements
* Train with a larger, more diverse dataset; add hand-crop augmentation; try the PyTorch CNN / ONNX route from `PROJECT_1_Blueprint.docx`.
* Real dynamic words with an LSTM/Transformer on landmark sequences (the sequence pipeline is already in place).
* Two-hand and body/face landmarks (MediaPipe Holistic) for real ASL grammar and facial expression.
* Per-user calibration, word prediction, more languages and other sign languages (BSL, PSL…).
* Real photos/videos for the Help Center (with proper licences).
* Packaging as a single Windows installer (PyInstaller).

## License
MIT – see `LICENSE`. Third-party libraries keep their own free licences. Check the licence of any dataset you download.
