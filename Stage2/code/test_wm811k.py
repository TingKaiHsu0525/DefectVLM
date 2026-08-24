import argparse
import os
import re
import json
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from PIL import Image

from model.openllama import OpenLLAMAPEFTModel

cli_parser = argparse.ArgumentParser()
cli_parser.add_argument("--output", default="eval_result_WM-811k.json",
                         help="Path to save the eval-result JSON")
cli_args = cli_parser.parse_args()

args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'imagebind_lora_rank': 16,
    'imagebind_lora_ckpt_dir': './ckpt/wm811k_stage1/50_epochs_lr=1e-4_rank=16/',
    'imagebind_lora_postfix': "_wm811_last",
    'imagebind_load_heads': True,
    'imagebind_load_postprocessors': True,
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'anomalygpt_ckpt_path': './ckpt/train_wm811k_normalize_1epoch/trainable_params.pt',
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
print(f"[!] Loaded trainable_params.pt. missing={len(missing)}, unexpected={len(unexpected)}")

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
    # Ensure output directory exists
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
 
with open(DATA2, "r") as f:
    test_data = json.load(f)

right, wrong = 0, 0
per_class_stats = {}
results = []
total_inference_time = 0.0
total_iou = 0.0
all_p_pred  = []
all_p_label = []

# Use regex to find the word inside single quotes
pattern = r"'(.*?)'"
num = 0
for i, sample in enumerate(test_data):
    img_path = os.path.join("../data", sample["img"])
    # print(f"\n[==== {i+1}/{len(test_data)} ====] Processing image: {img_path}")
    question = sample["prompt"]
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
    
    # anomaly map save path
    base = Path('../data').resolve()
    filename = os.path.basename(img_path.split('.')[0])
    for p in img_path:
        rel = Path(img_path).resolve().relative_to(base)
        filename = f"../view/wm811k/all/anomaly_map_{rel.stem}.png"

    # save anomaly map
    ab_map = anomaly_map[0, 1]  # (224,224)
    n_map = anomaly_map[0, 0]  # (224,224)
    save_anomaly_map(ab_map, filename, contour_thr=0.6)

    # IOU + Pixel-AUC
    ab_map = anomaly_map[0, 1].float()  # (224, 224)
    parts = sample["img"].split("/")    # ["WM-811k", "Center", "Center_288318.png"]
    mask_path = f"../data/WM-811k_Mask/{parts[1]}/{Path(img_path).stem}.png"
    gt_mask = torch.from_numpy(np.array(Image.open(mask_path))).float().to(ab_map.device)
    gt_mask = (gt_mask > 0).float()     # 255 → 1, 0 → 0

    pred_bin = (ab_map > 0.5).float()
    intersection = (pred_bin * gt_mask).sum()
    union = (pred_bin + gt_mask).clamp(max=1).sum()
    iou = (intersection / (union + 1e-6)).item()
    total_iou += iou

    all_p_pred.append(ab_map.detach().cpu().numpy().flatten())
    all_p_label.append(gt_mask.cpu().numpy().flatten().astype(np.uint8))

    
    # get defect type
    gt_pattern = re.search(pattern, gt_label.lower())
    gt_pattern = gt_pattern.group(1)

    pred_pattern = re.search(pattern, resp_lower)
    pred_pattern = pred_pattern.group(1)
    
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
        print(f"[✘] img={img_path}, expected: {gt_label}")
        print(f"Response: {resp}\n")

        # save wrong case images
        filename = f"../view/wm811k/wrong/anomaly_map_{rel.stem}.png"
        ab_map = anomaly_map[0, 1]  # (224,224)
        save_anomaly_map(ab_map, filename, contour_thr=0.6)

    # save results
    results.append({
        "img": sample["img"],
        "prompt": question,
        "label": sample["label"],
        "generated": resp,
        "is_correct": is_correct,
        "iou": iou,
    })
    
#  Print Results
accuracy = 100 * right / (right + wrong)
print(f"\nOverall Accuracy: {accuracy:.2f}%  (Total: {len(test_data)}, Right: {right}, Wrong: {wrong})")

avg_iou = total_iou / len(test_data) * 100
print(f"Average IOU: {avg_iou:.2f}")

p_auroc = roc_auc_score(
    np.concatenate(all_p_label),
    np.concatenate(all_p_pred)
) * 100
print(f"Pixel-AUROC: {p_auroc:.2f}%")

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