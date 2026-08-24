import argparse
import json
import re
from collections import defaultdict

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="mixedwm38_eval_result_test.json",
                     help="Path to the eval-result JSON produced by test_mixedwm38.py")
args = parser.parse_args()

# Class mapping table (label → Cxx)
label_to_c = {
    "Center+Edge_Loc+Scratch": "C24",
    "Center+Edge_Loc": "C10",
    "Center+Edge_Ring+Scratch": "C26",
    "Center+Edge_Ring": "C11",
    "Center+Loc+Edge_Loc+Scratch": "C35",
    "Center+Edge_Loc+Loc": "C23",
    "Center+Loc+Edge_ring+Scratch": "C36",
    "Center+Edge_Ring+Loc": "C25",
    "Center+Loc+Scratch": "C27",
    "Center+Loc": "C12",
    "Center+Scratch": "C13",
    "Center": "C2",
    "Dount+Edge_Loc+Scratch": "C29",
    "Dount+Edge_Loc": "C14",
    "Dount+Edge_Ring+Scratch": "C31",
    "Dount+Edge_Ring": "C15",
    "Dount+Loc+Edge_Loc+Scratch": "C37",
    "Dount+Edge_Loc+Loc": "C28",
    "Dount+Loc+Edge_Ring+Scratch": "C38",
    "Dount+Edge_Ring+Loc": "C30",
    "Dount+Loc+Scratch": "C32",
    "Dount+Loc": "C16",
    "Dount+Scratch": "C17",
    "Dount": "C3",
    "Edge_Loc": "C4",
    "Edge_Ring": "C5",
    "Edge_Loc+Loc+Scratch": "C33",
    "Edge_Loc+Loc": "C18",
    "Edge_Ring+Loc+Scratch": "C34",
    "Edge_Ring+Loc": "C20",
    "Loc+Scratch": "C22",
    "Loc": "C6",
    "Normal": "C1",
    "Edge_Loc+Scratch": "C19",
    "Edge_Ring+Scratch": "C21",
    "Scratch": "C8",
}

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

# Load inference result
with open(args.input, "r") as f:
    data = json.load(f)

# Count correct/total per defect type
stats = defaultdict(lambda: {"correct": 0, "total": 0})
for item in data:
    # Extract defect type 
    match = re.match(r"'(.+?)'", item["label"])
    if not match:
        continue
    defect_type = match.group(1)

    # skip unknown label types
    if defect_type not in label_to_c:
        continue  

    cname = label_to_c[defect_type]
    stats[cname]["total"] += 1
    if item["is_correct"]:
        stats[cname]["correct"] += 1

# Build accuracy per C1 ~ C38
accuracy_dict = {}
for defect, cname in label_to_c.items():
    correct = stats[cname]["correct"]
    total = stats[cname]["total"]
    acc = correct / total if total > 0 else 0.0
    accuracy_dict[cname] = (acc, correct, total)

# Build per-class IOU lists from JSON
iou_stats = defaultdict(list)
for item in data:
    match = re.match(r"'(.+?)'", item["label"])
    if not match:
        continue
    defect_type = match.group(1)
    if defect_type not in label_to_c:
        continue
    cname = label_to_c[defect_type]
    if "iou" in item:
        iou_stats[cname].append(item["iou"])

# mean IOU per class
iou_dict = {}
for defect, cname in label_to_c.items():
    ious = iou_stats[cname]
    iou_dict[cname] = (sum(ious) / len(ious) if ious else 0.0, ious)


def sort_key(cname):
    return int(cname[1:])

# average accuracy
def average_accuracy_range(accuracy_dict, c_start: int, c_end: int):
    """
    Compute macro and micro average accuracy over C{c_start}..C{c_end} inclusive.

    - Macro: mean of per-class accuracies, excluding classes with total == 0.
    - Micro: (sum of correct) / (sum of total) across the range.
    Returns: (macro, micro, counted_classes, counted_samples)
    """
    per_class_acc = []
    total_correct = 0
    total_seen = 0

    for i in range(c_start, c_end + 1):
        cname = f"C{i}"
        # accuracy_dict[cname] = (acc, correct, total) if present; else default to zeros
        acc, correct, total = accuracy_dict.get(cname, (0.0, 0, 0))
        if total > 0:
            per_class_acc.append(acc)
            total_correct += correct
            total_seen += total

    macro = (sum(per_class_acc) / len(per_class_acc)) if per_class_acc else 0.0
    micro = (total_correct / total_seen) if total_seen > 0 else 0.0
    return macro, micro, len(per_class_acc), total_seen

# average iou
def average_iou_range(iou_dict, c_start: int, c_end: int):
    per_class_iou = []
    all_ious = []
    for i in range(c_start, c_end + 1):
        cname = f"C{i}"
        mean_iou, ious = iou_dict.get(cname, (0.0, []))
        if ious:
            per_class_iou.append(mean_iou)
            all_ious.extend(ious)
    macro = sum(per_class_iou) / len(per_class_iou) if per_class_iou else 0.0
    micro = sum(all_ious) / len(all_ious) if all_ious else 0.0
    return macro, micro, len(per_class_iou), len(all_ious)


average_acc = 0
total_correct = 0
total_samples = 0
for cname in sorted(set(label_to_c.values()), key=sort_key):
    acc, correct, total = accuracy_dict[cname]
    print(f"{cname}: {acc:.2%} ({correct}/{total})")
    total_correct += correct
    total_samples += total
    average_acc += acc
average_acc /= len(set(label_to_c.values()))
print(f"Average accuracy: {average_acc:.2%} ({total_correct}/{total_samples:})")

all_iou = [item["iou"] for item in data if "iou" in item]
print(f"\nOverall Average IOU: {sum(all_iou)/len(all_iou):.4f}")


# per class acc
print("\nSingle Type(7):")
macro, micro, k_classes, n_samples = average_accuracy_range(accuracy_dict, 1, 9)
print(f"Macro average C1..C9 over {k_classes} classes: {macro:.2%}")
print(f"Micro average C1..C9 over {n_samples} samples: {micro:.2%}")
# iou 
macro_iou, micro_iou, _, _ = average_iou_range(iou_dict, 1, 9)
print(f"IOU C1..C9 — macro: {macro_iou:.4f}, micro: {micro_iou:.4f}")

print("\nTwo Mixed-Type(13):")
macro, micro, k_classes, n_samples = average_accuracy_range(accuracy_dict, 10, 22)
print(f"Macro average C10..C22 over {k_classes} classes: {macro:.2%}")
print(f"Micro average C10..C22 over {n_samples} samples: {micro:.2%}")
macro_iou, micro_iou, _, _ = average_iou_range(iou_dict, 10, 22)
print(f"IOU C10..C22 — macro: {macro_iou:.4f}, micro: {micro_iou:.4f}")

print("\nThree Mixed-Type(12):")
macro, micro, k_classes, n_samples = average_accuracy_range(accuracy_dict, 23, 34)
print(f"Macro average C23..C34 over {k_classes} classes: {macro:.2%}")
print(f"Micro average C23..C34 over {n_samples} samples: {micro:.2%}")
macro_iou, micro_iou, _, _ = average_iou_range(iou_dict, 23, 34)
print(f"IOU C23..C34 — macro: {macro_iou:.4f}, micro: {micro_iou:.4f}")


print("\nFour Mixed-Type(4):")
macro, micro, k_classes, n_samples = average_accuracy_range(accuracy_dict, 35, 38)
print(f"Macro average C35..C38 over {k_classes} classes: {macro:.2%}")
print(f"Micro average C35..C38 over {n_samples} samples: {micro:.2%}")
macro_iou, micro_iou, _, _ = average_iou_range(iou_dict, 35, 38)
print(f"IOU C35..C38 — macro: {macro_iou:.4f}, micro: {micro_iou:.4f}")


# BLEU on generated explanation text vs. ground-truth label text
all_preds = [item["generated"] for item in data if "generated" in item]
all_gts = [item["label"] for item in data if "generated" in item]
mean_bleu, std_bleu, n_bleu = compute_bleu_scores(all_preds, all_gts)

print("\n======== BLEU ========")
print(f"Samples: {n_bleu}")
print(f"Mean BLEU-2: {mean_bleu:.4f}")
print(f"Std  BLEU-2: {std_bleu:.4f}")