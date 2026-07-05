import os
import json
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image

from models.imagebind_model import ModalityType
import data


def label_to_reason_answer(label: str) -> str:
    # Convert a label into <answer>...</answer>\n<reason>...</reason> format
    import re
    s = (label or "").strip()
    if s == "":
        return "<answer></answer>"

    m = re.search(r"[.!?]", s)
    if m is None:
        answer = s
        reason = ""
    else:
        cut = m.end()
        answer = s[:cut].strip()
        reason = s[cut:].strip()

    if reason:
        return f"<answer>{answer}</answer>\n<reason>{reason}</reason>"
    return f"<answer>{answer}</answer>"


class WM811KImageBindDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        data_root: str,
        transform: Optional[Callable] = None,
        text_mode: str = "prompt+label",  # "label" | "prompt+label" | "class"

        # NOTE: backward-compatible options (only used when use_split=True)
        split: Optional[str] = None,      # "train" | "test" | None
        train_size: float = 0.8,
        random_seed: int = 42,
        use_split: bool = False,
    ):
        """
        Args:
            json_path: path to json list, each item has keys: img, prompt, label, class
                      If your json is already pre-split (train.json / val.json), keep use_split=False.
            data_root: root prefix to join with sample["img"]
                      e.g. if sample["img"] is "WM-811k/Center/xxx.png",
                      then data_root should be the directory that contains "WM-811k/"
            transform: should output either a tensor (C,H,W) or a list of tensors for multi-view
            text_mode: how to build text for contrastive learning
            use_split: if True, will split inside dataset using train_size/random_seed and split="train"/"test"
        """
        self.data_root = data_root
        self.transform = transform
        self.text_mode = text_mode

        with open(json_path, "r") as f:
            all_data = json.load(f)

        if not use_split:
            self.data = all_data
            return

        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            all_data, train_size=train_size, random_state=random_seed, shuffle=True
        )

        if split == "train":
            self.data = train_data
        elif split == "test":
            self.data = test_data
        else:
            raise ValueError("When use_split=True, split must be 'train' or 'test'.")

    def __len__(self):
        return len(self.data)

    def get_raw(self, index: int):
        sample = self.data[index]
        img_rel = sample["img"]
        img_path = os.path.join(self.data_root, img_rel)
        text_str = self._build_text(sample)
        return {
            "img_path": img_path,
            "prompt": sample.get("prompt", ""),
            "label": sample.get("label", ""),
            "class": sample.get("class", None),
            "text_str": text_str,
        }

    def _build_text(self, sample: dict) -> str:
        if self.text_mode == "label":
            s = (sample.get("label", "") or "").strip()
            import re
            m = re.search(r"[.!?]", s)
            if m is not None:
                s = s[:m.end()].strip()
            return s

        if self.text_mode == "class":
            return f"class_{int(sample.get('class', 0))}"

        prompt = (sample.get("prompt", "") or "").strip()
        label_fmt = label_to_reason_answer(sample.get("label", ""))

        prompt = prompt + "\nPlease answer in the following format:\n<answer>...</answer>\n<reason>...</reason>"
        return prompt + "\n" + label_fmt

    def __getitem__(self, index: int):
        sample = self.data[index]

        img_rel = sample["img"]
        img_path = os.path.join(self.data_root, img_rel)

        img = Image.open(img_path).convert("RGB")

        if self.transform is None:
            img_tensor = data.load_and_transform_vision_data([img_path], "cpu", to_tensor=True)[0]
            images_views = [img_tensor]
        else:
            out = self.transform(img)
            images_views = out if isinstance(out, list) else [out]

        text_str = self._build_text(sample)
        text_tensor = data.load_and_transform_text([text_str], "cpu")[0]

        class_id = int(sample.get("class", 0))
        return images_views, ModalityType.VISION, text_tensor, ModalityType.TEXT, class_id

    def collate(self, instances):
        data_a_list = []
        data_b_list = []
        labels = []

        for images_views, _, text_tensor, _, class_id in instances:
            data_a_list.append(images_views)
            data_b_list.append(text_tensor)
            labels.append(class_id)

        n_views = len(data_a_list[0])
        data_a = []
        for v in range(n_views):
            data_a.append(torch.stack([x[v] for x in data_a_list], dim=0))

        data_b = torch.stack(data_b_list, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        class_a = tuple([ModalityType.VISION] * len(instances))
        class_b = tuple([ModalityType.TEXT] * len(instances))

        return data_a, class_a, data_b, class_b, labels
