import argparse
import os
import re
import json
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score

from model.openllama import OpenLLAMAPEFTModel

cli_parser = argparse.ArgumentParser()
cli_parser.add_argument("--output", default="mixedwm38_eval_result.json",
                         help="Path to save the eval-result JSON")
cli_args = cli_parser.parse_args()

args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'imagebind_lora_rank': 16,
    'imagebind_lora_ckpt_dir': './ckpt/mixedwm38_stage1/50_epochs_lr=1e-4_rank=16/',
    'imagebind_lora_postfix': "_mixed_last",
    'imagebind_load_heads': True,
    'imagebind_load_postprocessors': True,
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_Mixedwm38_new_finetune_normalize/trainable_params.pt',
    'delta_ckpt_path': '../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}

model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location="cpu")
model.load_state_dict(delta_ckpt, strict=False)
trainable_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location="cpu")
missing, unexpected = model.load_state_dict(trainable_ckpt, strict=False)
model = model.eval().half().cuda()
print(f'[!] init the 7b model over ...')


def predict(input, image_path, max_length=512, top_p=0.1, temperature=1.0):  # top_p=0.1,  tempeaature=1.0
    response, pixel_output = model.generate({
        'prompt': input,
        'image_paths': [image_path],
        'audio_paths': [],
        'video_paths': [],
        'thermal_paths': [],
        'normal_img_paths': [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': []
    })
    return response, pixel_output

def save_anomaly_map(
    anomaly_map,
    file_name,
    contour_thr=0.7,          
    contour_top_percent=None  
):
    # Ensure directory exists
    out_dir = os.path.dirname(file_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    x = anomaly_map.detach().float().cpu()

    if x.ndim != 2:
        raise ValueError(f"Expected 2D map [H,W], but got shape={tuple(x.shape)}")

    x_np = x.numpy()

    # Decide contour threshold
    thr = contour_thr
    if contour_top_percent is not None:
        # Example: top 5% -> percentile 95
        thr = float(np.percentile(x_np, 100.0 - contour_top_percent))

    plt.figure()
    plt.imshow(x_np, cmap='hot')
    plt.colorbar()
    plt.title('Anomaly Map (abnormal prob)')

    # Draw contour line at threshold
    if thr is not None:
        plt.contour(x_np, levels=[thr], linewidths=2)

    plt.tight_layout()
    plt.savefig(file_name, dpi=150)
    plt.close()

# load test
DATA1 = "../data/test.json" # MixedWM-38 test set
DATA2 = "../data/WM-811k/test.json" # WM-811k test set

with open(DATA1, "r") as f:
    test_data = json.load(f)

right, wrong = 0, 0
per_class_stats = {}
results = []
total_inference_time = 0.0
total_iou = 0.0

complexity_p_pred  = {1: [], 2: [], 3: [], 4: []}
complexity_p_label = {1: [], 2: [], 3: [], 4: []}

# Use regex to find the word inside single quotes
pattern = r"'(.*?)'"
num = 0
for i, sample in enumerate(test_data):
    img_path = os.path.join("../data", sample["img"])
    # print(f"\n[==== {i+1}/{len(test_data)} ====] Processing image: {img_path}")
    question = sample["prompt"] # + "\nPlease answer in the following format:\n<answer>...</answer>\n<reason>...</reason>"
    gt_label = sample["label"].lower()
    gt_class = sample["class"]  
    # print(f"Evaluating sample {i+1}/{len(test_data)}: Image: {img_path}, Question: {question}, GT Label: {gt_label}")
    # breakpoint()

    # init per-class stats
    if gt_class not in per_class_stats:
        per_class_stats[gt_class] = {"right": 0, "wrong": 0}

    # predict
    t_start = time.time()
    resp, anomaly_map = predict(question, img_path)
    t_end = time.time()
    total_inference_time += (t_end - t_start)
    resp_lower = resp.lower()
    
    # get defect type
    gt_pattern = re.search(pattern, gt_label.lower())
    gt_pattern = gt_pattern.group(1)

    pred_pattern = re.search(pattern, resp_lower)
    pred_pattern = pred_pattern.group(1)

    # anomaly map save path
    base = Path('../data').resolve()
    rel = Path(img_path).resolve().relative_to(base)
    filename = f"../view1/mixedwm38/all/anomaly_map_{rel.stem}.png"
    ab_map = anomaly_map[0, 1]  # (224,224)
    n_map = anomaly_map[0, 0]  # (224,224)
    save_anomaly_map(ab_map, filename, contour_thr=0.6)

    # IOU
    ab_map = anomaly_map[0, 1]  # (224,224)
    img_name = Path(img_path).stem   # e.g., "36346"
    mask_path = f"../data/Masks/Image_{img_name}.npy"
    gt_mask = np.load(mask_path)  # shape (H, W, 1)
    gt_mask = gt_mask.squeeze(-1)  # shape (H, W)
    gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).float().to(ab_map.device)
    gt_mask = (gt_mask > 0).float()
    # gt_mask[gt_mask > 0], gt_mask[gt_mask <= 0] = 1, 0
    # print(f"gt_mask shape: {gt_mask.shape}, ab_map shape: {ab_map.shape}")
    # breakpoint()

    # 二值化 anomaly map
    pred_bin = (ab_map > 0.5).float()  # threshold 可調

    # 計算 IOU
    intersection = (pred_bin * gt_mask).sum()
    union = (pred_bin + gt_mask).clamp(max=1).sum()
    iou = (intersection / (union + 1e-6)).item()
    total_iou += iou


    # Pixel-AUC data collection
    n_types = len(gt_pattern.split('+'))
    complexity_p_pred[n_types].append(ab_map.detach().float().cpu().numpy().flatten())
    complexity_p_label[n_types].append(gt_mask.squeeze().cpu().numpy().flatten().astype(np.uint8))

    # by defect class
    if pred_pattern==gt_pattern:
        is_correct=True
    else:
        is_correct=False

    if is_correct:
        right += 1
        per_class_stats[gt_class]["right"] += 1
    else:
        wrong += 1
        per_class_stats[gt_class]["wrong"] += 1
        print(f"[✘] idx={i}, expected: {gt_label}")
        print(f"Response: {resp}\n")

        # save wrong case images
        filename = f"../view1/mixedwm38/wrong/anomaly_map_{rel.stem}.png"
        ab_map = anomaly_map[0, 1]  # (224,224)
        save_anomaly_map(ab_map, filename, contour_thr=0.6)

    # save results
    results.append({
        "img": sample["img"],
        "prompt": question,
        "label": sample["label"],
        "generated": resp,
        "is_correct": is_correct,
        "iou": iou
    })

#  Print Results
accuracy = 100 * right / (right + wrong)
print(f"\nOverall Accuracy: {accuracy:.2f}%  (Total: {len(test_data)}, Right: {right}, Wrong: {wrong})")

print("\nPixel-AUROC:")
p_auc_list = []
for n, name in [(1, 'Single'), (2, '2-Mixed'), (3, '3-Mixed'), (4, '4-Mixed')]:
    p_auroc = roc_auc_score(
        np.concatenate(complexity_p_label[n]),
        np.concatenate(complexity_p_pred[n])
    ) * 100
    p_auc_list.append(p_auroc)
    print(f"  {name}: {p_auroc:.4f}%")
print(f"  Average: {np.mean(p_auc_list):.4f}%")

# print average IOU
avg_iou = total_iou / len(test_data)
print(f"Average IOU: {avg_iou:.4f}")

avg_inference_time = total_inference_time / len(test_data) * 1000
print(f"Average Inference Time: {avg_inference_time:.4f} ms per sample")

print("\nPer-class Accuracy:")
for cls_id, stats in per_class_stats.items():
    total = stats["right"] + stats["wrong"]
    acc = 100 * stats["right"] / total if total > 0 else 0.0
    print(f"Class {cls_id}: {acc:.2f}%  (Right: {stats['right']}, Wrong: {stats['wrong']}, Total: {total})")

#  Save eval results
with open(cli_args.output, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"[✓] Saved eval results to {cli_args.output}")