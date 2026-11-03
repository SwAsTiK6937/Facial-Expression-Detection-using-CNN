"""Real-time facial expression recognition (webcam, video file, or image).

Usage:
    python webcam_test.py                     # default webcam
    python webcam_test.py --camera 1          # another camera
    python webcam_test.py --video clip.mp4    # fallback: a recorded video
    python webcam_test.py --image face.jpg    # fallback: a single image
    python webcam_test.py --model models/v1/fer_model.keras

Keys: q / Esc = quit, s = save screenshot, p = pause
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quieter TensorFlow logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np

from fer_model import EMOTIONS, IMG_SIZE, apply_correction, load_prior_correction, load_trained_model

# BGR colours per emotion for the probability bars
COLORS = {
    "angry": (60, 60, 230), "disgust": (60, 160, 60), "fear": (160, 60, 160),
    "happy": (40, 200, 240), "sad": (200, 120, 40), "surprise": (40, 140, 255),
    "neutral": (180, 180, 180),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    src.add_argument("--video", help="path to a video file instead of the webcam")
    src.add_argument("--image", help="path to a single image instead of the webcam")
    ap.add_argument("--model", help="model file (.keras or weights .h5). "
                    "Default: models/v1/fer_model.keras, else models/baseline/fer.h5")
    ap.add_argument("--detector", choices=["auto", "dnn", "haar"], default="auto",
                    help="face detector: dnn (robust, default if model files exist) or haar")
    ap.add_argument("--cascade", default="haarcascade_frontalface_default.xml")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="temporal smoothing 0..0.95 (0 = off). Reduces flicker.")
    ap.add_argument("--min-face", type=int, default=60, help="min face size in px")
    ap.add_argument("--output", help="save annotated image/video to this path")
    ap.add_argument("--no-correction", action="store_true",
                    help="disable the class-prior correction from metrics.json")
    ap.add_argument("--no-window", action="store_true", help="headless (for testing)")
    return ap.parse_args()


HERE = os.path.dirname(os.path.abspath(__file__))
DETECTORS = os.path.join(HERE, "models", "detectors")
YUNET_MODEL = os.path.join(DETECTORS, "face_detection_yunet_2023mar.onnx")

try:  # silence OpenCV's backend warnings
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def _local(path):
    """Resolve a path relative to this script if it isn't found as given."""
    if os.path.exists(path):
        return path
    for d in (DETECTORS, HERE):
        p = os.path.join(d, os.path.basename(path))
        if os.path.exists(p):
            return p
    return path


class FaceDetector:
    """YuNet CNN face detector (OpenCV >= 4.8, incl. 5.x); falls back to Haar cascade.

    The DNN detector handles head tilt, partial profiles, glasses and poor
    lighting far better than Haar, and produces far fewer false positives.
    """

    def __init__(self, kind="auto", cascade="haarcascade_frontalface_default.xml",
                 min_face=60, conf=0.6):
        self.min_face, self.conf, self.net, self.cascade = min_face, conf, None, None
        self.size = None
        weights = _local(YUNET_MODEL)
        if kind in ("auto", "dnn") and os.path.exists(weights) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.net = cv2.FaceDetectorYN.create(weights, "", (320, 320), conf, 0.3, 5000)
                self.name = "YuNet"
            except cv2.error as e:
                if kind == "dnn":
                    sys.exit(f"[error] could not load {YUNET_MODEL}: {e}")
                print(f"[warn] YuNet unavailable ({e}); using Haar cascade")
        elif kind == "dnn":
            sys.exit(f"[error] YuNet face model not found ({YUNET_MODEL}) or OpenCV < 4.8")
        if self.net is None:
            path = _local(cascade)
            if not os.path.exists(path):  # copy bundled with OpenCV
                path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self.cascade = cv2.CascadeClassifier(path)
            if self.cascade.empty():
                sys.exit(f"[error] could not load face cascade: {path}")
            self.name = "Haar"

    def __call__(self, frame, gray):
        H, W = gray.shape
        if self.net is not None:
            if self.size != (W, H):
                self.net.setInputSize((W, H))
                self.size = (W, H)
            _, det = self.net.detect(frame)
            boxes = [] if det is None else [tuple(d[:4]) for d in det]
        else:
            boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                                  minSize=(self.min_face, self.min_face))
        return [b for b in (square_box(b, W, H) for b in boxes)
                if b is not None and b[2] >= self.min_face]


def square_box(box, W, H):
    """Make the box square around its centre (FER2013 images are square, so a
    non-square crop would be distorted by the resize) and clip it to the frame."""
    x, y, w, h = (int(v) for v in box)
    side = max(w, h)
    cx, cy = x + w / 2, y + h / 2
    side = int(min(side, W, H))
    x0 = int(round(min(max(cx - side / 2, 0), W - side)))
    y0 = int(round(min(max(cy - side / 2, 0), H - side)))
    if side <= 0:
        return None
    return (x0, y0, side, side)


def preprocess(gray, box):
    """Crop face (rows = y..y+h, cols = x..x+w) and match training preprocessing."""
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return roi.astype("float32")[..., None] / 255.0  # same /255 scaling as training


class Smoother:
    """Exponential moving average of probabilities, matched to faces by position."""

    def __init__(self, alpha):
        self.alpha = alpha
        self.tracks = []  # list of (center, probs)

    def update(self, boxes, probs):
        if self.alpha <= 0:
            return probs
        new_tracks, out = [], []
        for box, p in zip(boxes, probs):
            c = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
            best, best_d = None, box[2] * 0.6  # must be within ~half a face width
            for tc, tp in self.tracks:
                d = np.linalg.norm(c - tc)
                if d < best_d:
                    best, best_d = tp, d
            sp = p if best is None else self.alpha * best + (1 - self.alpha) * p
            new_tracks.append((c, sp))
            out.append(sp)
        self.tracks = new_tracks
        return np.array(out)


def draw(frame, boxes, probs, fps, model_name):
    if frame.shape[0] < 360:  # upscale tiny frames so text/sidebar fit
        f = 360 / frame.shape[0]
        frame = cv2.resize(frame, None, fx=f, fy=f)
        boxes = [tuple(int(v * f) for v in b) for b in boxes]
    for (x, y, w, h), p in zip(boxes, probs):
        idx = int(np.argmax(p))
        label, conf = EMOTIONS[idx], float(p[idx])
        color = COLORS[label]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ty = max(y - 10, th + 10)
        cv2.rectangle(frame, (x, ty - th - 8), (x + tw + 8, ty + 4), color, -1)
        cv2.putText(frame, text, (x + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2, cv2.LINE_AA)

    status = f"faces: {len(boxes)}   model: {model_name}"
    if fps is not None:
        status = f"FPS: {fps:4.1f}   " + status
    cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 1, cv2.LINE_AA)
    return add_sidebar(frame, boxes, probs)


def add_sidebar(frame, boxes, probs, width=300):
    """Probability bars for the largest face, in a panel to the right of the video."""
    h = frame.shape[0]
    panel = np.full((h, width, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "Emotion probabilities", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if len(boxes):
        i = int(np.argmax([b[2] * b[3] for b in boxes]))
        p = probs[i]
        top = int(np.argmax(p))
        bw = width - 150
        for k, (name, v) in enumerate(zip(EMOTIONS, p)):
            yy = 60 + k * 32
            col = (255, 255, 255) if k == top else (170, 170, 170)
            cv2.putText(panel, name, (12, yy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        col, 2 if k == top else 1, cv2.LINE_AA)
            cv2.rectangle(panel, (95, yy), (95 + bw, yy + 18), (60, 60, 60), -1)
            cv2.rectangle(panel, (95, yy), (95 + int(bw * float(v)), yy + 18), COLORS[name], -1)
            cv2.putText(panel, f"{v * 100:5.1f}%", (100 + bw, yy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if len(boxes) > 1:
            cv2.putText(panel, "(showing largest face)", (12, 60 + 7 * 32 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "no face detected", (12, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (150, 150, 150), 1, cv2.LINE_AA)
    return np.hstack([frame, panel])


CORRECTION = None  # set in main() from metrics.json, if available


def predict(model, gray, boxes):
    if not boxes:
        return np.zeros((0, len(EMOTIONS)), dtype="float32")
    batch = np.stack([preprocess(gray, b) for b in boxes])
    # direct call is much faster than model.predict() for tiny batches
    return apply_correction(model(batch, training=False).numpy(), CORRECTION)


def run_image(args, model, model_name, detector):
    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"[error] could not read image: {args.image}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = detector(frame, gray)
    probs = predict(model, gray, boxes)
    for b, p in zip(boxes, probs):
        print(f"face at {b}: {EMOTIONS[int(np.argmax(p))]} "
              f"({', '.join(f'{e}={v:.2f}' for e, v in zip(EMOTIONS, p))})")
    if not boxes:
        print("no face detected")
    out = draw(frame, boxes, probs, None, model_name)
    if args.output:
        cv2.imwrite(args.output, out)
        print(f"saved {args.output}")
    if not args.no_window:
        cv2.imshow("Facial Expression Recognition", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def open_capture(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif sys.platform.startswith("win"):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)  # opens much faster on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        src = args.video or f"camera {args.camera}"
        sys.exit(f"[error] could not open {src}. Close other apps using the camera "
                 "(Zoom, Teams, browser) or try --camera 1, or use --video / --image.")
    return cap


def run_stream(args, model, model_name, detector):
    cap = open_capture(args)
    smoother = Smoother(min(max(args.smooth, 0.0), 0.95))
    writer, fails, paused, fps, last = None, 0, False, None, time.perf_counter()
    win = "Facial Expression Recognition (q to quit)"
    frame = None
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if args.video:
                        break  # end of file
                    fails += 1
                    if fails > 30:
                        print("[error] camera stopped returning frames")
                        break
                    continue
                fails = 0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = detector(frame, gray)
                probs = smoother.update(boxes, predict(model, gray, boxes))
                now = time.perf_counter()
                inst = 1.0 / max(now - last, 1e-6)
                fps = inst if fps is None else 0.9 * fps + 0.1 * inst
                last = now
                frame = draw(frame, boxes, probs, fps, model_name)
                if args.output:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(args.output,
                                                 cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                    writer.write(frame)
            if args.no_window:
                continue
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
            if key == ord("s"):
                name = time.strftime("screenshot_%H%M%S.png")
                cv2.imwrite(name, frame)
                print(f"saved {name}")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    global CORRECTION
    model, model_path = load_trained_model(args.model)
    model_name = os.path.basename(model_path)
    print(f"[info] loaded model: {model_path}")
    if not args.no_correction:
        CORRECTION = load_prior_correction(model_path)
        if CORRECTION is not None:
            print("[info] class-prior correction on (from metrics.json)")
    if model_name.lower() == "fer.h5":
        print("[warn] using the ORIGINAL weights (trained with a leaky split). "
              "Run train_fer_colab.ipynb to produce fer_model.keras.")
    model(np.zeros((1, IMG_SIZE, IMG_SIZE, 1), "float32"), training=False)  # warm-up
    detector = FaceDetector(args.detector, args.cascade, args.min_face)
    print(f"[info] face detector: {detector.name}")
    model_name += f" | {detector.name}"
    if args.image:
        run_image(args, model, model_name, detector)
    else:
        run_stream(args, model, model_name, detector)


if __name__ == "__main__":
    main()
