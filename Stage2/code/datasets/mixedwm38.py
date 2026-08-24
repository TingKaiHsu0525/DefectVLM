import os
import json
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
import re

import re


class MixedWM38Dataset(Dataset):
    def __init__(self, json_path: str):
        """
        Args:
            json_path (str): Path to train.json file
            img_root (str): Optional root dir for images if json has relative paths
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)


        # image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize(
            #         mean=(0.48145466, 0.4578275, 0.40821073),
            #         std=(0.26862954, 0.26130258, 0.27577711),
            #     ),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        # image path
        img_path = sample["img"]
        parts = img_path.split("/")
        img_path = "../data/"+img_path

        # mask path
        name, _= os.path.splitext(parts[1])  # name = "36346", ext = ".png"
        mask_path = "../data/Masks/Image_"+name+".npy"

        # print(f"Image path: {img_path}")
        # print(f"mask_path: {mask_path}")
        # breakpoint()

        # load image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        # load mask
        mask = np.load(mask_path)  # shape (H, W, 1)
        mask = mask.squeeze(-1)  # shape (H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        # print(f"img_tensor shape: {img_tensor.shape}")  # torch.Size([3, 224, 224])
        # print(f"mask_tensor shape: {mask_tensor.shape}")  # torch.Size([1, 224, 224])
        # breakpoint()

        # text conversation format
        question = sample["prompt"]
        answer = sample["label"]

        conversation = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer}  # full answer label 
        ]

        # class (int)
        class_id = sample["class"]    
        return conversation, img_tensor, class_id, img_path, mask_tensor

    def collate(self, instances):
        images = []
        texts = []
        classes = []
        img_paths = []
        masks = []

        for instance in instances:
            # instance: (conversation, img_tensor, class_id, img_path)
            texts.append(instance[0])
            images.append(instance[1])
            classes.append(instance[2])
            img_paths.append(instance[3])
            masks.append(instance[4])

        images = torch.stack(images, dim=0) # (B, 3, 224, 224)
        masks = torch.stack(masks, dim=0)   # (B, 1, 224, 224)

        return dict(
            images=images,
            texts=texts,
            classes=classes,
            img_paths=img_paths,
            mask=masks
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = MixedWM38Dataset("../data/train.json")
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate)
    num = 0
    for batch in loader:
        if num ==5:
            break
        # if batch["classes"] == [32]:
        print(batch["images"].shape)       # torch.Size([B, 3, 224, 224])
        print(batch["texts"][0])           # conversation format
        print(batch["classes"])            # list of class ids
        print(batch["img_paths"])          # original img paths
        print(batch["mask"].shape)         # torch.Size([B, 1, 224, 224])
            # break
        num += 1

