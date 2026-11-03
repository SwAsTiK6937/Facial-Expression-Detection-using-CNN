"""Evaluate the trained FER model on a FER2013 split and write a full metrics report.

Usage:
    python evaluate.py --csv data/fer2013/fer2013.csv                    # PrivateTest, models/v1/fer_model.keras
    python evaluate.py --csv data/fer2013/fer2013.csv --split PublicTest
    python evaluate.py --csv data/fer2013/fer2013.csv --model models/baseline/fer.h5     # original weights (leaky!)

Outputs (in --out, default "results/evaluation/"):
    evaluation_report.txt      human-readable report (raw and with class-prior correction)
    evaluation_metrics.json    every number, machine-readable
    confusion_matrix.png       counts and row-normalised matrices (with correction, if available)

Metrics:
    accuracy          correct / total
    precision (c)     TP / (TP + FP)       of the faces predicted c, how many really are c
    recall (c)        TP / (TP + FN)       of the faces that are c, how many were found
    F1 (c)            2PR / (P + R)        harmonic mean of precision and recall
    macro average     unweighted mean over the 7 classes (every class counts equally)
    weighted average  mean weighted by class support
    Cohen's kappa     agreement corrected for chance (0 = chance, 1 = perfect)
    top-2 accuracy    true class among the 2 most probable
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)

from fer_model import EMOTIONS, apply_correction, load_prior_correction, load_trained_model


def load_split(csv, split):
    df = pd.read_csv(csv)
    df = df[df["Usage"] == split]
    if df.empty:
        raise SystemExit(f"no rows with Usage == {split!r} in {csv}")
    x = np.stack([np.asarray(p.split(), dtype=np.uint8) for p in df["pixels"]])
    x = x.reshape(-1, 48, 48, 1).astype("float32") / 255.0  # same scaling as training
    return x, df["emotion"].to_numpy()


def metrics(y, probs):
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(7), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    top2 = np.mean([t in np.argsort(row)[-2:] for t, row in zip(y, probs)])
    cm = confusion_matrix(y, pred, labels=range(7))
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "top2_accuracy": round(float(top2), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y, pred)), 4),
        "macro": {"precision": round(float(mp), 4), "recall": round(float(mr), 4), "f1": round(float(mf), 4)},
        "weighted": {"precision": round(float(wp), 4), "recall": round(float(wr), 4), "f1": round(float(wf), 4)},
        "per_class": {e: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                          "f1": round(float(f[i]), 4), "support": int(s[i]), "predicted": int((pred == i).sum())}
                      for i, e in enumerate(EMOTIONS)},
        "confusion_matrix": cm.tolist(),
        "_report": classification_report(y, pred, labels=range(7), target_names=EMOTIONS, digits=3, zero_division=0),
    }


def plot_cm(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    for ax, data, fmt, sub in [(axes[0], cm, "d", "counts"), (axes[1], norm, ".2f", "row-normalised (recall on diagonal)")]:
        im = ax.imshow(data, cmap="Greens")
        ax.set_xticks(range(7), EMOTIONS, rotation=35, ha="right")
        ax.set_yticks(range(7), EMOTIONS)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(sub)
        thr = data.max() / 2
        for i in range(7):
            for j in range(7):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=10,
                        color="white" if v > thr else "#12312B")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="path to fer2013.csv")
    ap.add_argument("--split", default="PrivateTest", choices=["PrivateTest", "PublicTest", "Training"])
    ap.add_argument("--model", help="model file (default: models/v1/fer_model.keras, else models/baseline/fer.h5)")
    ap.add_argument("--out", default="results/evaluation")
    args = ap.parse_args()

    model, model_path = load_trained_model(args.model)
    corr = load_prior_correction(model_path)
    x, y = load_split(args.csv, args.split)
    raw = model.predict(x, batch_size=256, verbose=0)

    results = {"model": os.path.basename(model_path), "split": args.split, "raw": metrics(y, raw)}
    if corr is not None:
        results["corrected"] = metrics(y, apply_correction(raw, corr))
        results["correction_vector"] = [float(v) for v in corr]

    os.makedirs(args.out, exist_ok=True)
    lines = [f"Model: {results['model']}   Split: {args.split}   n = {len(y)}", ""]
    for key, label in [("raw", "WITHOUT class-prior correction"), ("corrected", "WITH class-prior correction")]:
        if key not in results:
            continue
        m = results[key]
        lines += [f"=== {label} ===",
                  f"accuracy {m['accuracy']:.4f}   top-2 accuracy {m['top2_accuracy']:.4f}   Cohen's kappa {m['cohen_kappa']:.4f}",
                  f"macro    P {m['macro']['precision']:.3f}  R {m['macro']['recall']:.3f}  F1 {m['macro']['f1']:.3f}",
                  f"weighted P {m['weighted']['precision']:.3f}  R {m['weighted']['recall']:.3f}  F1 {m['weighted']['f1']:.3f}",
                  "", m["_report"], "confusion matrix (rows = true, columns = predicted; order: " + ", ".join(EMOTIONS) + ")"]
        lines += ["  " + " ".join(f"{v:5d}" for v in row) for row in m["confusion_matrix"]]
        lines.append("")
    report = "\n".join(lines)
    open(os.path.join(args.out, "evaluation_report.txt"), "w", encoding="utf-8").write(report)
    for k in ("raw", "corrected"):
        if k in results:
            results[k].pop("_report")
    json.dump(results, open(os.path.join(args.out, "evaluation_metrics.json"), "w"), indent=2)
    best = "corrected" if "corrected" in results else "raw"
    plot_cm(results[best]["confusion_matrix"],
            f"{results['model']} on FER2013 {args.split} ({'with' if best == 'corrected' else 'without'} prior correction)",
            os.path.join(args.out, "confusion_matrix.png"))
    print(report)
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
