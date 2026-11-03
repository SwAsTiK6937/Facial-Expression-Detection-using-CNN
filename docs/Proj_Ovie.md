# Project Overview: Real-Time Facial Expression Recognition with CNNs

**Author:** Swastik Pandey, B.Tech CSE, SRMIST
**Courses:** Computer Vision · Neural Networks and Deep Learning (final project)
**Last updated:** 1 November 2026

---

## 1. One-paragraph summary

A real-time system that detects faces in a webcam feed and classifies each face into one of 7 expressions (angry, disgust, fear, happy, sad, surprise, neutral). It started from an open-source CNN project that reported **81.3%** test accuracy on FER2013. Auditing that project showed the number was inflated by **data leakage**: the dataset was oversampled *before* the train/test split, so **62.1% of the "test" images had an exact copy in the training set**. The project fixes the evaluation (official splits, duplicate removal, train-only class weighting), retrains the model, and reports an honest **57.6%** accuracy (macro-F1 **0.520**) on the untouched FER2013 PrivateTest set. It also rebuilds the real-time demo: a modern face detector (YuNet), correct preprocessing, an inference-time class-prior correction, temporal smoothing and a one-click Windows launcher.

---

## 2. Credit and starting point

- **Baseline repository:** https://github.com/shivamsingh96/Facial-Expression-Detection-using-CNN-Real-Time-Recognition-with-Webcam
- From the baseline: the CNN architecture, the original notebook (`Real-Time Facial Expression Detection & Recognition using CNN.ipynb`, kept unchanged), `fer.h5`, the Haar cascade file and the original deck.
- **New work in this project:** the leakage audit and fix, the leak-free training notebook, the analysis (diagnosis, prior correction), the rewritten demo, the face detector upgrade, the launcher, documentation and both presentation decks.

---

## 3. Repository contents

| File | Purpose | Status |
|---|---|---|
| `webcam_test.py` | Real-time demo (webcam, video file or image) | Rewritten |
| `fer_model.py` | Model definition, version-independent loading, class-prior correction | New |
| `run_demo.bat` | One-click Windows setup and launch | New |
| `notebooks/train_fer_colab.ipynb` | Leak-free training and evaluation notebook (now **v2**, see §8.4) | New |
| `requirements.txt` | Python dependencies | New |
| `evaluate.py` | Full evaluation: accuracy, precision, recall, F1, kappa, top-2, confusion matrix | New |
| `results/evaluation/` | Evaluation report, metrics JSON and confusion-matrix image from `evaluate.py` | New |
| `README.md` | Project README with credit and changes | Rewritten |
| `Proj_Ovie.md` | This overview | New |
| `models/v1/fer_model.keras` | Retrained leak-free model (v1, baseline architecture) | New (trained on Colab) |
| `models/v1/fer.weights.h5` | Weights-only copy of the same model | New |
| `metrics.json` | Test metrics + class-prior correction vector | New |
| `models/v1/training_curves.png`, `confusion_matrix.png` | Training and evaluation plots | New |
| `models/v1/` | v1 model, metrics and plots (single copy) | New |
| `models/detectors/face_detection_yunet_2023mar.onnx` | YuNet face detector (OpenCV Zoo, 232 KB) | New |
| `models/detectors/haarcascade_frontalface_default.xml` | Haar face detector (fallback) | Baseline |
| `models/baseline/fer.h5` | Original baseline weights (trained with the leaky protocol) | Baseline, fallback only |
| `Facial Expression Recognition.json` | Original Keras 2 model JSON (cannot load in Keras 3) | Baseline, unused |
| `notebooks/Real-Time Facial Expression Detection & Recognition using CNN.ipynb` | Original notebook | Baseline, unchanged |
| `data/fer2013/fer2013.csv` | Dataset (35,887 images) | Downloaded |

---

## 4. System pipeline (Computer Vision view)

```
Webcam frame (BGR)
  → 1. Face detection        YuNet CNN (OpenCV FaceDetectorYN); Haar cascade fallback
  → 2. Square crop           box squared around its centre, clipped to the frame
  → 3. Grayscale + resize    48×48, INTER_AREA
  → 4. Normalise             pixels ÷ 255 (identical to training)
  → 5. CNN classifier        7-way softmax
  → 6. Prior correction      p(c) · w_c^(−0.75), renormalised
  → 7. Temporal smoothing    per-face EMA, α = 0.6
  → 8. Overlay               box, label + confidence, probability sidebar, FPS
```

### 4.1 Face detection
| | Haar cascade (Viola–Jones, 2001) | YuNet (used) |
|---|---|---|
| Type | Haar-like features + AdaBoost + attention cascade | Lightweight CNN, ONNX (232 KB) |
| Output | Box | Box + 5 landmarks + confidence |
| Weakness | Tilted faces, harsh lighting | – |
| Role | Automatic fallback | Default (`--detector auto`) |

**Robustness test** (same portrait degraded 7 ways, min face 30 px):

| Condition | YuNet | Haar |
|---|---|---|
| Original | Detected | Detected |
| Rotated 20° | Detected | Detected |
| Rotated 30° | Detected | **Missed** |
| Brightness 25% | Detected | Detected |
| Downscaled to 256 px | Detected | Detected |
| Low contrast | Detected | Detected |
| Gaussian blur | Detected | Detected |
| Blank frame (no face) | No false positive | No false positive |

YuNet: 7/7 face conditions; Haar: 6/7. A small controlled test, not a benchmark.

### 4.2 Preprocessing decisions
- **Square crop:** FER2013 images are square. YuNet returned a 90×113 box in testing; resizing that straight to 48×48 would squash the face.
- **INTER_AREA** downsampling avoids aliasing when shrinking a ~100 px face to 48 px.
- **Same scaling as training** (÷255), so there is no train/test input mismatch.
- **Clipping:** boxes near the frame edge are shifted inside rather than cut short.

### 4.3 Real-time engineering
- One batched model call per frame (all faces at once) instead of `model.predict()` per face.
- Per-face exponential moving average of probabilities (α = 0.6) to stop label flicker.
- DirectShow camera backend on Windows for fast start-up.
- FPS counter and per-class probability sidebar for the largest face.
- Fallback modes: `--video file.mp4` and `--image file.jpg`.

---

## 5. Dataset: FER2013

- 35,887 grayscale face images, 48×48 px, 7 classes.
- Official splits (`Usage` column): **Training 28,709 · PublicTest 3,589 · PrivateTest 3,589**.
- Class counts: angry 4,953 · **disgust 547** · fear 5,121 · **happy 8,989** · sad 6,077 · surprise 4,002 · neutral 6,198. Disgust has 16× fewer images than happy.
- Human agreement on FER2013 is about **65 ± 5%** (Goodfellow et al., 2013), so the labels are noisy.

---

## 6. The data-leakage finding (central result)

**Baseline protocol (wrong order):** whole dataset → randomly duplicate images until every class has 8,989 → random 90/10 split.
Copies of the same image land on both sides of the split.

**Measured by re-creating the protocol:**
- **62.1%** of "test" rows have an exact copy in the training set.
- Per class: **disgust 100%** (each image copied ~16×), **happy 0%** (never copied), angry ~75%, surprise ~84%, fear ~69%, neutral ~53%, sad ~51%.
- Evidence in the baseline's own report: disgust, the hardest FER2013 class, scored **0.99 precision / 1.00 recall**, while happy (no copies) scored 0.80 recall.
- The original `fer.h5` weights score **89.0%** on the official PrivateTest set, but only because most of those images were in its training data.

**Fix (leak-free protocol):**
1. Official splits: Training → train, PublicTest → validation (model selection), PrivateTest → test (used once).
2. **703** training images that are exact (MD5) duplicates of validation/test images removed. Final sizes: 28,006 / 3,589 / 3,589.
3. Class imbalance handled with a class-weighted loss on the training split only (no duplication).
4. Early stopping, checkpoints and tuning use validation only.

---

## 7. Model (Neural Networks view)

### 7.1 Architecture (baseline network, used for the reported results)

| Block | Layers (input 48×48×1) | Output | Params |
|---|---|---|---|
| 1 | Conv 3×3 ×32 (valid) · BN · ReLU · Dropout 0.25 | 46×46×32 | 448 |
| 2 | Conv 3×3 ×64 (same) · BN · ReLU · MaxPool | 23×23×64 | 18,752 |
| 3 | Conv 3×3 ×64 (valid) · BN · ReLU · Dropout 0.25 | 21×21×64 | 37,184 |
| 4 | Conv 3×3 ×128 (same) · BN · ReLU · MaxPool | 10×10×128 | 74,368 |
| 5 | Conv 3×3 ×128 (valid) · BN · ReLU · MaxPool | 4×4×128 | 148,096 |
| – | Flatten | 2,048 | 0 |
| – | Dense 250 · ReLU · Dropout 0.5 | 250 | 512,250 |
| – | Dense 7 · Softmax | 7 | 1,757 |
| | **Total** | | **792,855** (792,023 trainable) |

Notable: **65%** of all parameters (512,250) are in the single Dense-250 layer; the conv feature extractor is only ~279K parameters.

Design choices: BatchNorm before the activation (as in Ioffe & Szegedy, 2015), ReLU, dropout, mixed same/valid padding, softmax + categorical cross-entropy.

### 7.2 Training setup (v1, the run behind the reported numbers)

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 64 |
| Epochs | 80 (all 80 ran; early stopping never triggered) |
| LR schedule | ReduceLROnPlateau, ×0.5 after 4 flat epochs, min 1e-5 |
| Early stopping | val accuracy, patience 12, restore best |
| Augmentation | horizontal flip, ±18° rotation, 10% translation, 10% zoom |
| Class weights | balanced (disgust ≈ 9.41×, happy ≈ 0.57×) |
| Hardware | Google Colab GPU |

### 7.3 Training dynamics
- **Collapse at epochs 17–23:** training accuracy dropped and validation accuracy fell to ~5% at epoch 23 before recovering after the LR was halved. Likely cause: large gradients from the 9.4× disgust class weight.
- **Validation above training** throughout: augmentation and dropout are active only during training. This is not leakage.
- **Never converged:** still improving at epoch 80.

### 7.4 Diagnosis: underfitting
- Training accuracy on clean (un-augmented) images: **58.2%** vs validation **56.9%**. A 1.3-point gap: the model isn't memorising, it isn't learning enough.
- Likely causes: limited capacity (small conv extractor), strong regularisation (augmentation + three dropout layers), gradient spikes from the large class weight.

### 7.5 Class-prior correction (inference time)
Training with class weight w_c makes the model over-predict rare classes. At inference the probabilities are multiplied by w_c^(−a) and renormalised; **a = 0.75** was chosen on the validation set (maximising macro-F1), then applied once to the test set.

Correction vector (stored in `metrics.json`, order angry→neutral):
`[0.64189, 0.12188, 0.65414, 1.0, 0.74009, 0.53978, 0.75555]`

| Test set (PrivateTest) | Before | After |
|---|---|---|
| Disgust predictions | 214 | 45 |
| Disgust precision | 0.20 | 0.44 |
| Accuracy | 56.0% | **57.6%** |
| Macro-F1 | 0.496 | **0.520** |

The demo applies it automatically when `metrics.json` sits next to `fer_model.keras` (in `models/v1/`) (disable with `--no-correction`).

---

## 8. Results

### 8.1 Headline (PrivateTest, 3,589 images, evaluated once)
| Metric | Raw | With prior correction |
|---|---|---|
| Accuracy | 56.03% | **57.62%** |
| Macro-F1 | 0.496 | **0.520** |

Other splits (raw): PublicTest 56.87%, Training (clean) 58.19%.

### 8.2 Per-class (raw, before correction)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.470 | 0.350 | 0.401 | 491 |
| disgust | 0.196 | 0.764 | 0.312 | 55 |
| fear | 0.302 | 0.165 | 0.213 | 528 |
| happy | 0.877 | 0.833 | **0.854** | 879 |
| sad | 0.411 | 0.382 | 0.396 | 594 |
| surprise | 0.659 | 0.776 | 0.713 | 416 |
| neutral | 0.507 | 0.684 | 0.582 | 626 |

**Most confused pairs (true → predicted):** sad → neutral 30% · fear → sad 22% · fear → surprise 19% · angry → sad 18%. Fear recall is only 16%.

**Live demo expectations:** happy and surprise are reliable; neutral is okay; angry, sad and fear are weak.

### 8.3 Full evaluation metrics (`evaluate.py`)
Reproduce every number with:
```
python evaluate.py --csv data/fer2013/fer2013.csv     # writes results/evaluation/evaluation_report.txt, evaluation_metrics.json, confusion_matrix.png
```

| Metric (PrivateTest, n = 3,589) | Raw | With prior correction |
|---|---|---|
| Accuracy | 56.03% | **57.62%** |
| Top-2 accuracy | 76.51% | **77.65%** |
| Cohen's kappa | 0.473 | **0.486** |
| Macro precision / recall / F1 | 0.489 / 0.565 / 0.496 | **0.527 / 0.525 / 0.520** |
| Weighted precision / recall / F1 | 0.559 / 0.560 / 0.550 | **0.560 / 0.576 / 0.562** |

Per-class, **with** prior correction:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.480 | 0.407 | 0.441 | 491 |
| disgust | 0.444 | 0.364 | 0.400 | 55 |
| fear | 0.314 | 0.172 | 0.223 | 528 |
| happy | 0.828 | 0.857 | 0.842 | 879 |
| sad | 0.419 | 0.446 | 0.432 | 594 |
| surprise | 0.696 | 0.736 | 0.715 | 416 |
| neutral | 0.507 | 0.692 | 0.585 | 626 |

Correction trade-off: disgust precision 0.20 → 0.44 but recall 0.76 → 0.36 (51% of disgust faces are then predicted as angry). Largest confusion after correction: sad → neutral (29%).

Metric definitions: precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R); macro = unweighted mean over the 7 classes; weighted = by class support; Cohen's kappa = (pₒ − pₑ)/(1 − pₑ), agreement beyond chance; top-2 = true class among the two most probable outputs.

### 8.4 Notebook v2 (implemented, NOT yet evaluated)
`train_fer_colab.ipynb` now trains an improved configuration by default:
- VGG-style network: 4 blocks × 2 convs (64-128-256-512), BN, global average pooling, Dense 256; ~4.82M parameters (`FER_ARCH=baseline` trains the original network instead)
- **Square-root** balanced class weights (disgust ~3× instead of ~9.4×)
- Gradient clipping (`clipnorm=1.0`), up to 100 epochs, early stopping patience 15, LR patience 5
- Correction strength tuned automatically on validation and written to `metrics.json`

A short CPU check was inconclusive, so **no accuracy is claimed for v2**. If it is run: abort if validation accuracy is below ~50% at epoch 30, and keep `models/v1/` if the new test accuracy is below 0.576.

---

## 9. Literature survey (22 papers) and comparison

Accuracies are **as reported by the authors** (or in the comparison table of [17]); test splits, extra data and ensembling differ, so they are not strictly comparable. Two values = single model / ensemble.

| Ref | Authors | Venue | Approach | FER2013 |
|---|---|---|---|---|
| [1] | Goodfellow et al. | arXiv 2013 | Introduced FER2013; human accuracy | 65 ± 5% (human) |
| [2] | Tang | ICML Workshop 2013 | CNN + linear SVM loss; challenge winner | 71.2% |
| [3] | Ionescu, Popescu & Grozea | ICML Workshop 2013 | Bag of visual words + local SVM | 67.48% |
| [4] | Zhang, Luo, Loy & Tang | ICCV 2015 | Siamese CNN + HoG; extra training data | 75.1% |
| [5] | Mollahosseini, Chan & Mahoor | WACV 2016 | Inception-style CNN, 7 training sets | 66.4% |
| [6] | Kim et al. | CVPRW 2016 | Ensemble: aligned + non-aligned CNNs | 73.73% |
| [7] | Liu, Zhang & Pan | Cyberworlds 2016 | CNN ensemble study | 62.44% |
| [8] | Pramerdorfer & Kampel | arXiv 2016 | VGG, ResNet, Inception; 8-CNN ensemble | 72.7 / 75.2% |
| [9] | Giannopoulos et al. | Springer SIST 2018 | GoogLeNet and AlexNet case study | 65.2% |
| [10] | Arriaga, Valdenegro-Toro & Plöger | ESANN 2019 | mini-Xception, 60K params, real-time | 66% |
| [11] | Georgescu, Ionescu & Popescu | IEEE Access 2019 | CNN + bag-of-words features, local SVM | 75.42% |
| [12] | Li & Deng | IEEE TAC 2020 | Survey of deep FER methods | survey |
| [13] | Khanzada, Bai & Celepcikay | arXiv 2020 | Ensemble + transfer learning | 75.8% |
| [14] | Pham, Vu & Tran | ICPR 2020 | Residual Masking Network | 74.14 / 76.82% |
| [15] | Minaee, Minaei & Abdolrashidi | Sensors 2021 | Attentional CNN + spatial transformer | 70.02% |
| [16] | Shi et al. | arXiv 2021 | Amend Representation Module, ResNet-18 | 71.38% |
| [17] | Khaireddin & Chen | arXiv 2021 | Tuned VGGNet | 73.28% |
| [18] | Fard & Mahoor | IEEE Access 2022 | Ad-Corre correlation loss, Xception | 72.03% |
| [19] | Pecoraro, Basile & Bono | Information 2022 | ResNet-34 + channel self-attention | 74.42% |
| [20] | Vignesh et al. | IJIT 2023 | VGG-19 + U-Net segmentation layers | 75.97% |
| [21] | El Boudouri & Bohi | IEEE MMSP 2023 | EmoNeXt: ConvNeXt + STN + SE blocks | 76.12% |
| [22] | Roy et al. | arXiv 2024 | ResEmoteNet: CNN + SE + residual | 79.79% |

**Source-confidence notes:** [22] is an arXiv preprint whose 79.79% has not been independently verified. The 73.73% for [6] comes from a paper citing it. The 62.44% for [7] may be a single-network figure rather than the ensemble's.

### 9.1 Trends
- **2013:** handcrafted features (67.48% [3]) vs. the first CNNs (71.2% [2]).
- **2015–16:** deeper CNNs and ensembles (73.73% [6], 75.2% [8]); extra data (75.1% [4]).
- **2017–20:** efficiency and hybrids: mini-Xception 66% with 60K params [10]; CNN + handcrafted 75.42% [11].
- **2021–24:** attention and modern backbones: 73.28% [17], 74.42% [19], 76.12% [21], 79.79% [22].
- Ensembles add 2–3 points (72.7 → 75.2% [8]; 74.14 → 76.82% [14]).

### 9.2 Our algorithm vs. past algorithm families
| Family | Representative | Key idea | Scale | FER2013 |
|---|---|---|---|---|
| Handcrafted | Ionescu et al. [3] | BoVW + local SVM | no deep network | 67.48% |
| Early CNN | Tang [2] | CNN with SVM loss | CNN ensemble | 71.2% |
| Compact, real-time | Arriaga et al. [10] | Depthwise-separable mini-Xception | 60K params | 66% |
| Deep single CNN | Khaireddin & Chen [17] | Tuned VGGNet | VGG-scale | 73.28% |
| Attention | Pecoraro et al. [19] | Channel self-attention in ResNet-34 | ResNet-34 | 74.42% |
| Ensemble | Pham et al. [14] | ResMaskingNet + 6 CNNs | 7 networks | 76.82% |
| **This project** | Ours | 5-conv CNN, class weights + prior correction, leak-free | 0.79M, single | **57.6%** |

**Why ours scores lower:** no face alignment, pretraining or ensembling; a small conv extractor (~279K params); the model underfits.
**What this project adds:** an audited, leak-free protocol; inference-time class-prior correction; a real-time CPU demo.

---

## 10. Engineering fixes to the baseline demo

| Problem | Effect | Fix |
|---|---|---|
| Model saved as Keras 2 JSON | Crashed on TensorFlow ≥ 2.16 (Keras 3) | Architecture defined in code (`fer_model.py`) |
| Crop `gray[y:y+w, x:x+h]` | Wrong crop for non-square boxes | `gray[y:y+h, x:x+w]` |
| `cv2.destroyAllWindows` without `()` | Window never closed | Called in a `finally` block |
| No camera → `continue` forever | Program hangs | Clear error; exits after 30 failed reads |
| `model.predict()` per face per frame | Slow, console spam | One batched direct call per frame |
| Haar cascade only | Misses tilted faces | YuNet detector, Haar fallback |
| Non-square boxes resized to 48×48 | Distorted faces | Square crop around the face centre |
| Files resolved from the working directory | Failed when launched elsewhere | Paths resolved relative to the script |
| OpenCV 5 dropped the Caffe importer | First DNN detector choice would crash | Switched to YuNet (ONNX), works on OpenCV 4.8+ and 5.x |
| TensorFlow has no Windows wheels for Python 3.14 | `pip install` fails | Launcher requires Python 3.10–3.13 (3.12 recommended) |

---

## 11. How to run

### 11.1 Demo (Windows, easiest)
1. Install **Python 3.12** from python.org (tick *Add python.exe to PATH*). Not 3.14.
2. Double-click **`run_demo.bat`**. The first run creates `%USERPROFILE%\.venvs\fer` and installs TensorFlow + OpenCV (~600 MB, 5–10 min); later runs start immediately.
3. The console should show `class-prior correction on (from metrics.json)` and `face detector: YuNet`.

### 11.2 Options
```
run_demo.bat                      default webcam
run_demo.bat --camera 1           another camera
run_demo.bat --video demo.mp4     play a recorded clip (backup for live demos)
run_demo.bat --image face.jpg     single image
run_demo.bat --output demo.mp4    record the annotated stream
run_demo.bat --detector haar      force the Haar detector
run_demo.bat --smooth 0           disable temporal smoothing
run_demo.bat --no-correction      disable the class-prior correction
run_demo.bat --model path         use a specific model file
```
Keys: **q**/Esc quit · **p** pause · **s** screenshot.

Model loading order: `models/v1/fer_model.keras` → `models/v1/fer.weights.h5` → `models/baseline/fer.h5` (incompatible files are skipped automatically).

### 11.3 Training (Google Colab)
1. Put `fer2013.csv` in Google Drive.
2. Open `notebooks/train_fer_colab.ipynb` → Runtime → Change runtime type → T4 GPU.
3. Set `CSV_PATH`, then Runtime → Run all.
4. Download `fer_model.keras`, `fer.weights.h5`, `metrics.json` and the plots into `models/v2/` (keep `models/v1/` as the backup) and update the paths in `fer_model.py`.

---

## 12. Presentation decks
- **Computer Vision** (15 slides): https://claude.ai/artifact/WpomGiCUJsxT9G6499k1yW
  Pipeline, detection (Haar vs YuNet), robustness test, preprocessing, real-time engineering, fixes, results with confusion matrix, failure analysis, live demo, next steps.
- **Neural Networks and Deep Learning** (24 slides): https://claude.ai/artifact/GAjdPcCQeeVZfrezP1drod
  Problem, 22-paper literature survey, trends, the red flag, leakage bug and fix, architecture, training, dynamics, diagnosis, prior correction, results, comparison with past algorithms, honest comparison, lessons, future work, references.

Both can be exported to PowerPoint or PDF from the deck's share/export menu. CV slide 9 has a placeholder for a demo screenshot (press **s** during the demo).

---

## 13. Limitations
- Underfitting baseline architecture; v2 is unevaluated.
- Weak classes: fear (F1 0.21), disgust (0.31), sad (0.40), angry (0.40).
- No face alignment; sensitive to strong backlight, extreme pose and occlusion.
- FER2013 label noise caps achievable accuracy (human ≈ 65 ± 5%).
- The detector robustness test is a single-image controlled test, not a benchmark.
- Literature accuracies are reported values, not reproduced.

## 14. Future work
- Evaluate v2 (VGG-style, sqrt class weights, gradient clipping).
- Face alignment using YuNet's 5 landmarks; CLAHE lighting normalisation.
- Transfer learning from face-pretrained networks; ensembles; test-time augmentation.
- Multi-face tracking with stable IDs; video-level (temporal) modelling.

---

## 15. Viva / Q&A preparation

**Why is your accuracy lower than the original 81%?**
The original's test set overlapped 62% with its training set (oversampling before splitting). Its own weights score 89% on the official test set only because they trained on most of it. 57.6% is a real estimate on unseen faces.

**Why is validation accuracy above training accuracy?**
Augmentation and dropout are active only during training. On clean images, training accuracy is 58.2% vs 56.9% validation.

**Is the model overfitting?**
No, it underfits: a 1.3-point train/validation gap on clean data. More capacity and less regularisation are needed, not more dropout.

**What is the class-prior correction?**
Class weighting biases the model toward rare classes. Multiplying outputs by w_c^(−0.75), with the exponent chosen on validation only, cut disgust false alarms from 214 to 45 and raised accuracy 1.6 points without retraining.

**Why YuNet instead of Haar?**
In the robustness test it found a 30°-tilted face Haar missed, with no false positives on a blank frame. It also returns landmarks, which enables face alignment later.

**Why is fear so bad?**
Recall 16%: fear spreads across sad, surprise and neutral, which share cues (raised brows, open mouth). It is also a hard class in published work.

**Why not use the best published model?**
The project's focus is a correct evaluation of a compact real-time CNN and an honest diagnosis; a deeper VGG-style network is implemented (v2) as the next step.

---

## 16. Numbers cheat-sheet
| Fact | Value |
|---|---|
| Dataset size | 35,887 images, 48×48, 7 classes |
| Splits (after dedup) | 28,006 / 3,589 / 3,589 |
| Duplicates removed | 703 |
| Leaked test rows (baseline protocol) | 62.1% (disgust 100%, happy 0%) |
| Baseline reported accuracy | 81.31% (leaky) |
| Baseline weights on real test set | 89.0% (trained on it) |
| Our test accuracy | 57.6% (56.0% raw) |
| Our macro-F1 | 0.520 (0.496 raw) |
| Clean train vs validation | 58.2% vs 56.9% |
| Parameters | 792,855 (65% in Dense-250) |
| Correction exponent | a = 0.75 |
| Disgust predictions before/after | 214 → 45 |
| Human agreement | ≈ 65 ± 5% |
| Best surveyed (as reported) | 79.79% (ResEmoteNet, arXiv 2024) |
