import logging
import torch
import data

from models import imagebind_model
from models.imagebind_model import ModalityType, load_module
from models import lora as LoRA

logging.basicConfig(level=logging.INFO, force=True)


lora = True
linear_probing = False
device = "cpu"  # "cuda:0" if torch.cuda.is_available() else "cpu"
load_head_post_proc_finetuned = True

assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. "

if lora and not load_head_post_proc_finetuned:
    # Hack: adjust lora_factor to the `max batch size used during training / temperature` to compensate missing norm
    lora_factor = 12 / 0.07
else:
    # This assumes proper loading of all params but results in shift from original dist in case of LoRA
    lora_factor = 1

text_list=["bird",
           "car",
           "dog3",
           "dog5",
           "dog8",
           "grey_sloth_plushie"]
image_paths=[".assets/bird_image.jpg",
             ".assets/car_image.jpg",
             ".assets/dog3.jpg",
             ".assets/dog5.jpg",
             ".assets/dog8.jpg",
             ".assets/grey_sloth_plushie.jpg"]
audio_paths=[".assets/bird_audio.wav",
             ".assets/car_audio.wav",
             ".assets/dog_audio.wav"]

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
if lora:
    model.modality_trunks.update(
        LoRA.apply_lora_modality_trunks(model.modality_trunks, rank=4,
                                        # layer_idxs={ModalityType.TEXT: [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                        #             ModalityType.VISION: [0, 1, 2, 3, 4, 5, 6, 7, 8]},
                                        modality_names=[ModalityType.TEXT, ModalityType.VISION]))

    # Load LoRA params if found
    LoRA.load_lora_modality_trunks(model.modality_trunks,
                                   checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")

    if load_head_post_proc_finetuned:
        # Load postprocessors & heads
        load_module(model.modality_postprocessors, module_name="postprocessors",
                    checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")
        load_module(model.modality_heads, module_name="heads",
                    checkpoint_dir=".checkpoints/lora/550_epochs_lora", postfix="_dreambooth_last")
elif linear_probing:
    # Load heads
    load_module(model.modality_heads, module_name="heads",
                checkpoint_dir="./.checkpoints/lora/500_epochs_lp", postfix="_dreambooth_last")

model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device, to_tensor=True),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)
    
def print_topk(sim: torch.Tensor, row_names, col_names, k=3, title=""):
    # Print top-k matches for each row item
    print(f"\n=== {title} (top-{k}) ===")
    for i in range(sim.shape[0]):
        vals, idxs = torch.topk(sim[i], k=min(k, sim.shape[1]), dim=-1)
        pairs = []
        for v, j in zip(vals.tolist(), idxs.tolist()):
            pairs.append(f"{col_names[j]}:{v:.4f}")
        print(f"{row_names[i]} -> " + ", ".join(pairs))


def normalize_for_softmax(x: torch.Tensor):
    # Stabilize softmax for readability
    x = x - x.max(dim=-1, keepdim=True).values
    return torch.softmax(x, dim=-1)


# Similarity logits (dot product)
sim_vt = embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T
sim_at = embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T
sim_va = embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T

# Apply scaling if you want to keep original behavior
scale = (lora_factor if lora else 1.0)
sim_vt = sim_vt * scale
sim_at = sim_at * scale

# print raw similarity 
print_topk(sim_vt, image_paths, text_list, k=3, title="Vision x Text (raw similarity)")
print_topk(sim_at, audio_paths, text_list, k=3, title="Audio x Text (raw similarity)")
print_topk(sim_va, image_paths, audio_paths, k=3, title="Vision x Audio (raw similarity)")

# probability-like output, use softmax then top-k
prob_vt = normalize_for_softmax(sim_vt)
prob_at = normalize_for_softmax(sim_at)
prob_va = normalize_for_softmax(sim_va)

print_topk(prob_vt, image_paths, text_list, k=3, title="Vision x Text (softmax prob)")
print_topk(prob_at, audio_paths, text_list, k=3, title="Audio x Text (softmax prob)")
print_topk(prob_va, image_paths, audio_paths, k=3, title="Vision x Audio (softmax prob)")


