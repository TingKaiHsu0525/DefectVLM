import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

CLASS_NAMES = {
    0: "Center",
    1: "Donut",
    2: "Edge-Loc",
    3: "Edge-Ring",
    4: "Loc",
    5: "Scratch",
}
DIR_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

# flexible mapping: normalize predicted text to class id (-1 = unknown)
_PRED_MAP = {
    "center":    0,
    "donut":     1,
    "dount":     1,
    "edge_loc":  2, "edge-loc":  2, "edge loc":  2,
    "edge_ring": 3, "edge-ring": 3, "edge ring": 3,
    "loc":       4,
    "scratch":   5,
}

def _parse_pred(generated: str) -> int:
    """Extract class id from generated text. Returns -1 if unknown."""
    m = re.match(r"'(.+?)'", generated)
    if m:
        key = m.group(1).strip().lower()
        return _PRED_MAP.get(key, -1)
    return -1


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def compute_bleu_scores(preds, gts):
    smoothie = SmoothingFunction().method4
    scores = []
    for p, g in zip(preds, gts):
        p_n = normalize_text(p)
        g_n = normalize_text(g)
        if not g_n.strip():
            continue
        score = sentence_bleu(
            [g_n.split()],
            p_n.split(),
            weights=(0.5, 0.5),
            smoothing_function=smoothie,
        )
        scores.append(score)
    if not scores:
        return 0.0, 0.0, 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def compute_accuracy(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    n_cls = len(CLASS_NAMES)
    total_right = 0
    total_wrong = 0
    class_right = defaultdict(int)
    class_wrong = defaultdict(int)

    # confusion[true][pred] = count  (pred == -1 means unknown/unparseable)
    confusion = defaultdict(lambda: defaultdict(int))

    for item in data:
        class_dir = item["img"].split("/")[1]
        true_id   = DIR_TO_CLASS[class_dir]
        pred_id   = _parse_pred(item["generated"])
        correct   = item["is_correct"]

        confusion[true_id][pred_id] += 1

        if correct:
            class_right[true_id] += 1
            total_right += 1
        else:
            class_wrong[true_id] += 1
            total_wrong += 1

    # ── Accuracy ──────────────────────────────────────────────────────────
    total = total_right + total_wrong
    overall_acc = 100.0 * total_right / total if total > 0 else 0.0

    print(f"Overall Accuracy: {overall_acc:.2f}%  "
          f"(Total: {total}, Right: {total_right}, Wrong: {total_wrong})")
    print()
    print("Per-class Accuracy:")
    for class_id in sorted(CLASS_NAMES.keys()):
        r = class_right[class_id]
        w = class_wrong[class_id]
        t = r + w
        acc = 100.0 * r / t if t > 0 else 0.0
        print(f"Class {class_id} ({CLASS_NAMES[class_id]:10s}): {acc:.2f}%  "
              f"(Right: {r}, Wrong: {w}, Total: {t})")

    # ── Confusion Matrix (text) ───────────────────────────────────────────
    print()
    print("Confusion Matrix (rows=True, cols=Predicted):")

    col_labels = [CLASS_NAMES[i] for i in range(n_cls)] + ["Unknown"]
    col_ids    = list(range(n_cls)) + [-1]

    header_pad = 12
    col_w = 10
    print(" " * header_pad + "".join(f"{name:>{col_w}}" for name in col_labels))
    print(" " * header_pad + "-" * (col_w * len(col_labels)))

    for true_id in range(n_cls):
        row_label = f"{CLASS_NAMES[true_id]:>{header_pad - 2}} |"
        row = ""
        for pred_id in col_ids:
            cnt = confusion[true_id][pred_id]
            row += f"{cnt:>{col_w}}"
        print(row_label + row)

    # ── Confusion Matrix (image, 6×6 only, no Unknown) ───────────────────
    labels = [CLASS_NAMES[i] for i in range(n_cls)]
    cm = np.array([[confusion[t][p] for p in range(n_cls)] for t in range(n_cls)])

    # normalize per row to show recall rate
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sum

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13)

    # annotate each cell with count only
    thresh = 0.5
    for t in range(n_cls):
        for p in range(n_cls):
            val   = cm[t, p]
            norm  = cm_norm[t, p]
            color = "white" if norm > thresh else "black"
            ax.text(p, t, f"{val}",
                    ha="center", va="center", fontsize=11, color=color)

    plt.tight_layout()

    out_path = Path(json_path).with_suffix(".confusion.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix image saved to: {out_path}")

    # ── BLEU on generated explanation text vs. ground-truth label text ────
    all_preds = [item["generated"] for item in data if "generated" in item]
    all_gts = [item["label"] for item in data if "generated" in item]
    mean_bleu, std_bleu, n_bleu = compute_bleu_scores(all_preds, all_gts)

    print("\n======== BLEU ========")
    print(f"Samples: {n_bleu}")
    print(f"Mean BLEU-2: {mean_bleu:.4f}")
    print(f"Std  BLEU-2: {std_bleu:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval_result_WM-811k.json",
                         help="Path to the eval-result JSON produced by test_wm811k.py")
    args = parser.parse_args()
    compute_accuracy(args.input)
