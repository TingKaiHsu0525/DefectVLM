import os
import json
import random
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class MixedWM38Dataset(Dataset):
    def __init__(
        self,
        json_path: str,
        task_probs=None,
        seed: int = 42,
        use_label_v2: bool = True,
        fixed_task= None,
    ):
        """
        Args:
            json_path: Path to json file
            task_probs: Dict[str, float], e.g. {"type":0.7, "count":0.2, "reason":0.1}
            seed: random seed for sampling tasks
            use_label_v2: if True, use label_v2 to build multitask QA
            fixed_task: if not None, force a task in {"type","count","reason"} (useful for eval)
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        # English comments only
        self.use_label_v2 = use_label_v2
        self.fixed_task = fixed_task

        # English comments only
        self.rng = random.Random(seed)

        # English comments only
        self.task_probs = task_probs or {
            "type": 0.7,
            "count": 0.1,
            "reason": 0.2,
        }

        # English comments only
        self.type_prompts = [
            "What is the defect in this wafer map?",
            "Identify the defect type in this wafer map.",
            "Which defect pattern is shown in this wafer map?",
        ]
        # IMPORTANT: your count is the number of defect patterns in the quoted label (split by '+')
        self.count_prompts = [
            "How many defect patterns are present in this wafer map?",
            "How many defect types are included in this wafer map?",
            "Count the number of defect patterns in this wafer map.",
        ]
        self.reason_prompts = [
            "Describe this defect pattern.",
            "Explain the defect characteristics in this wafer map.",
            "What does this defect look like? Describe it.",
        ]

    def __len__(self):
        return len(self.data)

    def _choose_task(self) -> str:
        # English comments only
        if self.fixed_task in ("type", "count", "reason"):
            return self.fixed_task

        tasks = list(self.task_probs.keys())
        weights = [float(self.task_probs[t]) for t in tasks]
        s = sum(weights)
        if s <= 0:
            return "type"
        weights = [w / s for w in weights]

        r = self.rng.random()
        acc = 0.0
        for t, w in zip(tasks, weights):
            acc += w
            if r <= acc:
                return t
        return tasks[-1]

    def _sample_prompt(self, task: str) -> str:
        # English comments only
        if task == "type":
            return self.rng.choice(self.type_prompts)
        if task == "count":
            return self.rng.choice(self.count_prompts)
        if task == "reason":
            return self.rng.choice(self.reason_prompts)
        return self.rng.choice(self.type_prompts)

    def _build_answer_from_v2(self, sample: dict, task: str) -> str:
        # English comments only
        v2 = sample.get("label_v2", {}) if isinstance(sample.get("label_v2", {}), dict) else {}
        defect_label = str(v2.get("defect_label", "")).strip()
        count = v2.get("count", 0)
        reason = str(v2.get("reason", "")).strip()

        if task == "type":
            # Keep it short and consistent
            return defect_label if defect_label else str(sample.get("label", "")).strip()

        if task == "count":
            # Return only the number to reduce hallucination
            try:
                return str(int(count))
            except Exception:
                return "0"

        if task == "reason":
            # Return only the description part
            return reason if reason else ""

        return str(sample.get("label", "")).strip()

    def __getitem__(self, index):
        sample = self.data[index]

        # image path
        img_path = sample["img"]
        parts = img_path.split("/")
        img_path_full = "../data/" + img_path

        # mask path
        name, _ = os.path.splitext(parts[1])
        mask_path = "../data/Masks/Image_" + name + ".npy"

        # load image
        img = Image.open(img_path_full).convert("RGB")
        img_tensor = self.transform(img)

        # load mask
        mask = np.load(mask_path)
        mask = mask.squeeze(-1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        # choose task + build QA
        if self.use_label_v2 and isinstance(sample.get("label_v2", None), dict):
            task = self._choose_task()
            question = self._sample_prompt(task)
            answer = self._build_answer_from_v2(sample, task)
        else:
            # fallback to original single-task data
            task = "type"
            question = sample.get("prompt", "What is the defect in this wafer map?")
            answer = sample.get("label", "")

        # text conversation format
        conversation = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer}
        ]

        # class (int)
        class_id = sample["class"]

        # Return format unchanged
        return conversation, img_tensor, class_id, img_path_full, mask_tensor

    def collate(self, instances):
        images = []
        texts = []
        classes = []
        img_paths = []
        masks = []

        for instance in instances:
            texts.append(instance[0])
            images.append(instance[1])
            classes.append(instance[2])
            img_paths.append(instance[3])
            masks.append(instance[4])

        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)

        return dict(
            images=images,
            texts=texts,
            classes=classes,
            img_paths=img_paths,
            mask=masks
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = MixedWM38Dataset(
        "../data/train_mixed.json",
        task_probs={"type": 0.7, "count": 0.1, "reason": 0.2},
        use_label_v2=True,
        fixed_task=None,  # set to "type"/"count"/"reason" for debugging
    )

    print(f"Dataset size: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate)

    for i, batch in enumerate(loader):
        if i == 3:
            break
        print(batch["img_paths"])
        print(batch["images"].shape)
        print(batch["texts"][0])
        print(batch["texts"][1])
        print(batch["classes"])
        print(batch["mask"].shape)
        print("-----")
