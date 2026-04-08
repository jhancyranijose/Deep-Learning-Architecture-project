#!/usr/bin/env python3
"""
evaluate.py (fixed)
CPU-only evaluation + confusion matrix + reports/plots.
Uses CLI args --model and --data (no hardcoded dr_model.h5).

Saves into:
  outputs/checkpoint/<run>_checkpoint/
  outputs/plots/<run>_plots/
  outputs/reports/<run>_reports/
Never overwrites: if folders exist, adds _1, _2, ...

Example:
  python evaluate.py --data ./data/DR_Image --model dr_run4_model.keras --run-tag dr_run4
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, must be before TF import

import re
import json
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
from datetime import datetime

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Dataset root folder, e.g. ./data/DR_Image")
    p.add_argument("--model", type=str, required=True, help="Model file to evaluate, e.g. dr_run4_model.keras")
    p.add_argument("--outputs", type=str, default="outputs", help="Outputs base folder")
    p.add_argument("--run-tag", type=str, default=None, help="e.g. dr_run4. If omitted, auto dr_runN.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--img", type=int, default=224)
    p.add_argument("--batch", type=int, default=20)
    return p.parse_args()


def ensure_dirs(base: Path):
    (base / "checkpoint").mkdir(parents=True, exist_ok=True)
    (base / "plots").mkdir(parents=True, exist_ok=True)
    (base / "reports").mkdir(parents=True, exist_ok=True)
    return base / "checkpoint", base / "plots", base / "reports"


def next_run_id(checkpoint_dir: Path) -> int:
    pat = re.compile(r"^dr_run(\d+)_checkpoint")
    nums = []
    for p in checkpoint_dir.iterdir():
        if p.is_dir():
            m = pat.match(p.name)
            if m:
                nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def make_run_folders(run_tag: str, ckpt_root: Path, plots_root: Path, reports_root: Path):
    ckpt = ckpt_root / f"{run_tag}_checkpoint"
    plots = plots_root / f"{run_tag}_plots"
    reports = reports_root / f"{run_tag}_reports"

    # Avoid overwriting
    suffix = 0
    base_ckpt, base_plots, base_reports = ckpt, plots, reports
    while ckpt.exists() or plots.exists() or reports.exists():
        suffix += 1
        ckpt = Path(str(base_ckpt) + f"_{suffix}")
        plots = Path(str(base_plots) + f"_{suffix}")
        reports = Path(str(base_reports) + f"_{suffix}")

    ckpt.mkdir(parents=True, exist_ok=False)
    plots.mkdir(parents=True, exist_ok=False)
    reports.mkdir(parents=True, exist_ok=False)
    return ckpt, plots, reports


def load_dataset(data_path: Path):
    print("Loading dataset...")

    imgpaths, labels = [], []
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"Found classes: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        for f in class_dir.iterdir():
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                imgpaths.append(str(f))
                labels.append(class_dir.name)

    df = pd.DataFrame({"Paths": imgpaths, "Labels": labels})
    print(f"Total images loaded: {len(df)}\n")
    return df


def make_test_generator(df, img_size, batch_size, seed):
    train, testval = train_test_split(df, test_size=0.2, shuffle=True, random_state=seed)
    valid, test = train_test_split(testval, test_size=0.5, shuffle=True, random_state=seed)

    print(f"Train set: {train.shape[0]} images")
    print(f"Validation set: {valid.shape[0]} images")
    print(f"Test set: {test.shape[0]} images\n")

    # Match dr5.py style (no rescale)
    gen = ImageDataGenerator()
    test_gen = gen.flow_from_dataframe(
        test,
        x_col="Paths",
        y_col="Labels",
        target_size=(img_size, img_size),
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )
    return test_gen


def main():
    args = parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)
    outputs_base = Path(args.outputs)

    if not data_path.exists():
        raise SystemExit(f"ERROR: Data path not found: {data_path}")
    if not model_path.exists():
        raise SystemExit(f"ERROR: Model file not found: {model_path}")

    ckpt_root, plots_root, reports_root = ensure_dirs(outputs_base)

    if args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = f"dr_run{next_run_id(ckpt_root)}"

    ckpt_dir, plots_dir, reports_dir = make_run_folders(run_tag, ckpt_root, plots_root, reports_root)

    print(f"\nRun tag: {run_tag}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Plots dir:      {plots_dir}")
    print(f"Reports dir:    {reports_dir}\n")

    df = load_dataset(data_path)
    test_gen = make_test_generator(df, args.img, args.batch, args.seed)

    print(f"Loading model from: {model_path}")  # ✅ IMPORTANT
    model = keras.models.load_model(model_path, compile=False)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
)

    # Copy model into checkpoint folder
    dst = ckpt_dir / model_path.name
    dst.write_bytes(model_path.read_bytes())
    print(f"Saved checkpoint copy: {dst}")

    print("\nEvaluating model on TEST set...")
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"Test  - Loss: {loss:.4f}, Accuracy: {acc:.4f}\n")

    print("Generating predictions for confusion matrix (CPU-only)...")
    preds = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    classes = list(test_gen.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(reports_dir / "confusion_matrix.csv")

    rep = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    (reports_dir / "classification_report.txt").write_text(rep)

    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title("Confusion Matrix - Test Set")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center",
                 color="white" if cm[i, j] > thresh else "red")

    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    summary = {
        "run": run_tag,
        "model_path_used": str(model_path),
        "data_path": str(data_path),
        "img_size": [args.img, args.img],
        "batch_size": args.batch,
        "seed": args.seed,
        "metrics": {"test_loss": float(loss), "test_acc": float(acc)},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "class_indices": test_gen.class_indices,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Saved confusion matrix plot:", plots_dir / "confusion_matrix.png")
    print("Saved report:", reports_dir / "classification_report.txt")
    print("Saved summary:", reports_dir / "summary.json")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()