# Facial Expression Recognition with a CNN: leak-free evaluation & real-time demo

Real-time recognition of 7 facial expressions (angry, disgust, fear, happy, sad, surprise, neutral) from a webcam,
using a CNN trained on FER2013 and OpenCV's YuNet face detector.

## Credit / baseline

This project builds on
[shivamsingh96/Facial-Expression-Detection-using-CNN-Real-Time-Recognition-with-Webcam](https://github.com/shivamsingh96/Facial-Expression-Detection-using-CNN-Real-Time-Recognition-with-Webcam).
The CNN architecture and the original notebook
(`Real-Time Facial Expression Detection & Recognition using CNN.ipynb`) come from that repository and are kept
unchanged for reference. Everything listed under *What I changed* is new work.

## What I changed

### 1. Found and fixed data leakage in the evaluation
The baseline balanced the classes by **randomly duplicating images over the whole dataset and only then splitting it
into train/test**. Copies of the same image therefore landed on both sides of the split.

Re-creating that protocol on FER2013 shows that **about 62% of the "test" images have an exact copy in the
training set**: 100% for *disgust* (547 images duplicated about 16×), 0% for *happy* (never duplicated).
That explains the baseline's suspicious per-class results: *disgust*, the hardest FER2013 class,
scored 0.99 precision / 1.00 recall, while *happy* had no copies and scored a more ordinary 0.80 recall.
The reported **81.3% accuracy measures memorisation, not generalisation.**

The fix (`train_fer_colab.ipynb`):
- the official FER2013 splits: `Training` → train, `PublicTest` → validation, `PrivateTest` → test (used once)
- exact-duplicate images shared between training and val/test removed from training
- class imbalance handled with a **class-weighted loss on the training split** instead of duplicating images
- data augmentation (flip, rotation, shift, zoom), early stopping, LR-on-plateau, best-checkpoint restore
- accuracy, macro-F1, per-class report and a normalised confusion matrix on the untouched test set

### 2. Fixed the real-time demo (`webcam_test.py`)
| Problem in the original | Fix |
|---|---|
| Crashed on current TensorFlow: the Keras 2.12 JSON can't be loaded by Keras 3 | architecture defined in code (`fer_model.py`); loads the new `.keras` model or the old `fer.h5` |
| `cv2.destroyAllWindows` without `()`, so it was never called | fixed, with cleanup in `finally` |
| Face crop used `[y:y+w, x:x+h]` (width/height swapped) | `[y:y+h, x:x+w]` |
| Infinite loop when no camera was available | clear error message; exits after repeated read failures |
| `model.predict()` per face per frame (slow, log spam) | one batched direct call per frame |
| Labels flicker frame-to-frame | per-face temporal smoothing (`--smooth`) |
| Haar cascade misses tilted faces and fires on background clutter | **YuNet** CNN face detector (`face_detection_yunet_2023mar.onnx`, OpenCV Zoo); Haar kept as fallback (`--detector haar`) |
| Non-square face boxes were squashed when resized to 48×48 | boxes made square around the face centre, clipped to the frame |
| Files found only if run from the repo folder | paths resolved relative to the script |
| — | confidence %, probability sidebar, FPS, pause/screenshot keys, `--video` / `--image` fallback modes, DirectShow backend on Windows |

### 3. Documentation corrections
The baseline README said the dense layer has 200 units (the code uses 250) and reported 81.39% (the notebook shows 81.31%).
It also described putting BatchNorm *before* the activation as unusual. That is the placement proposed in the original
BatchNorm paper (Ioffe & Szegedy, 2015), so it isn't presented as a contribution here.

## Results

| | Baseline (original protocol) | This project |
|---|---|---|
| Test set | random 10% of oversampled data | official FER2013 PrivateTest (3,589 images) |
| Train/test overlap | ~62% of test images seen in training | none |
| Test accuracy | 81.3% (inflated) | **56.0%** (57.6% with class-prior correction) |
| Macro-F1 | 0.81 (inflated) | **0.496** (0.520 with correction) |

Measured with the baseline architecture (notebook v1, 80 epochs). Notebook v2 trains a VGG-style network with
milder class weights and gradient clipping. Its numbers are written to `metrics.json` when you run it.

Note: the original `fer.h5` scores 89% on PrivateTest, but only because it was trained on those images.

For context, FER2013 human-level accuracy is reported at about 65 ± 5%, and the best published single models
reach the mid-70s.

## How to run

### Demo (Windows), easiest way
Install **Python 3.12** from python.org (tick *Add python.exe to PATH*). TensorFlow has no Windows build for Python 3.14,
so don't use the newest version. Then double-click **`run_demo.bat`**. The first run installs everything
(about 5–10 min); later runs start straight away. Arguments pass through, e.g. `run_demo.bat --camera 1`.

### Demo (Windows), manual
```bat
:: one-time setup: create the virtual environment OUTSIDE OneDrive so it doesn't sync thousands of files
py -3.12 -m venv C:\venvs\fer
C:\venvs\fer\Scripts\python.exe -m pip install --upgrade pip
C:\venvs\fer\Scripts\python.exe -m pip install -r requirements.txt

:: run (from this folder)
C:\venvs\fer\Scripts\python.exe webcam_test.py
```
Options: `--camera 1` (other camera), `--video clip.mp4` or `--image face.jpg` (no webcam needed),
`--output out.mp4` (record), `--smooth 0` (no smoothing), `--model path`.
Keys: **q**/Esc quit · **p** pause · **s** screenshot.

The demo uses `models/v1/fer_model.keras` (retrained, leak-free) if present, otherwise the baseline `models/baseline/fer.h5`.

### Training (Google Colab, about 15–25 min on a T4 GPU)
1. Get `fer2013.csv` (Kaggle FER2013 challenge CSV) into Google Drive.
2. Open `notebooks/train_fer_colab.ipynb` in Colab → *Runtime → Change runtime type → T4 GPU*.
3. Set `CSV_PATH` in section 2, then *Runtime → Run all*.
4. Download `fer_model.keras` (and `metrics.json`, the plots) and put them in `models/v1/` (or a new `models/v2/`, then update the paths in `fer_model.py`).

## Project structure
```
├── webcam_test.py          real-time demo (rewritten)
├── fer_model.py            model definition + version-independent loading (new)
├── evaluate.py             full evaluation metrics (new)
├── run_demo.bat            one-click Windows setup + launch
├── requirements.txt
├── models/
│   ├── v1/                 retrained leak-free model: fer_model.keras, fer.weights.h5,
│   │                       metrics.json, training_curves.png, confusion_matrix.png
│   ├── baseline/           fer.h5 (leaky protocol, fallback only) + Keras 2 model JSON (no longer needed)
│   └── detectors/          YuNet face detector (OpenCV Zoo, MIT licence) + Haar cascade (fallback)
├── notebooks/
│   ├── train_fer_colab.ipynb                         leak-free training & evaluation (new)
│   └── Real-Time Facial Expression ... CNN.ipynb     baseline notebook, unchanged
├── data/fer2013/           FER2013 dataset (not in git)
├── results/evaluation/     output of evaluate.py
└── docs/                   project overview (Proj_Ovie.md) + presentation
```
