import json
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import open_clip
from tqdm import tqdm

from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, label2paths, transform, image_root):
        self.paths = []
        self.labels = []
        self.image_root = image_root
        for label, paths in label2paths.items():
            for path in paths:
                self.paths.append(path)
                self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_root, self.paths[idx]))
        if self.transform:
            img = self.transform(img)
        return img


def generate_options(labels2paths, image_root, output_folder, dataset_name,
                     top_k=4, prompt_prefix = "an image of a {}",
                     model_name="ViT-L-14", pretrained="laion2b_s32b_b82k", batchsize=512, workers=4):

    if os.path.exists(f"{output_folder}/{dataset_name}-{model_name}-{pretrained}-{top_k}.json"):
        print("Options already exist. Loading")
        label_options = json.load(open(f"{output_folder}/{dataset_name}-{model_name}-{pretrained}-{top_k}.json"))
        return label_options

    print("Creating options")

    device = "cuda"
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    dataset = CustomDataset(labels2paths, transform, image_root)
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=workers)

    labels = list(labels2paths.keys())
    prompts = [prompt_prefix.format(c) for c in labels]
    text_input = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_input)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)

    cosine_sim = []
    for i, (img) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating embeddings"):
        img = img.to(device)
        with torch.no_grad():
            image_feat = model.encode_image(img)

        image_feat /= image_feat.norm(dim=-1, keepdim=True)

        cos_sim = (image_feat @ text_feat.T).cpu().tolist()  # calculate cosine sim

        cosine_sim.extend(cos_sim)

    label_options = defaultdict(list)

    for i, cos in enumerate(cosine_sim):
        # get each image and groundtruth
        groundtruth = dataset.labels[i]
        gt_idx = labels.index(groundtruth)

        indices = np.argsort(cos)[-top_k:]
        # assure gt is always there and first
        if gt_idx not in indices:
            indices = [gt_idx] + [idx for idx in indices[1:]]
        else:
            indices = [gt_idx] + [idx for idx in indices if idx != gt_idx]
        options = [labels[idx] for idx in indices]
        label_options[groundtruth].append({
            "image": dataset.paths[i],
            "options": options,
            "groundtruth": groundtruth
        })

    os.makedirs(output_folder, exist_ok=True)
    json.dump(label_options, open(f"{output_folder}/{dataset_name}-{model_name}-{pretrained}-{top_k}.json", "w"))
    return label_options


def generate_options_idx(labels2paths, image_root, output_folder, dataset_name, prompt_prefix="an image of a {}",
                     model_name="ViT-L-14", pretrained="laion2b_s32b_b82k", batchsize=512, workers=4):
    if os.path.exists(f"{output_folder}/{dataset_name}-{model_name}-{pretrained}-idx.json"):
        print("Options already exist. Loading")
        label_options = json.load(open(f"{output_folder}/{dataset_name}-{model_name}-{pretrained}-idx.json"))
        return label_options

    print("Creating options")

    device = "cuda"
    model, _, transform = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    dataset = CustomDataset(labels2paths, transform, image_root)
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=workers)

    labels = list(labels2paths.keys())
    prompts = [prompt_prefix.format(c) for c in labels]
    text_input = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_input)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)

    cosine_sim = []
    for i, (img) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating embeddings"):
        img = img.to(device)
        with torch.no_grad():
            image_feat = model.encode_image(img)

        image_feat /= image_feat.norm(dim=-1, keepdim=True)

        cos_sim = (image_feat @ text_feat.T).cpu().tolist()  # calculate cosine sim

        cosine_sim.extend(cos_sim)

    label_options = defaultdict(list)

    for i, cos in enumerate(cosine_sim):
        # get each image and groundtruth
        groundtruth = dataset.labels[i]
        gt_idx = labels.index(groundtruth)

        indices = np.argsort(cos).tolist()
        # assure gt is always there and first
        # if gt_idx not in indices:
        #     indices = [gt_idx] + [idx for idx in indices[1:]]
        # else:
        indices = [idx for idx in indices if idx != gt_idx] + [gt_idx]
        label_options[gt_idx].append({
            "image": dataset.paths[i],
            "options": indices,
            "groundtruth": gt_idx
        })

    os.makedirs(output_folder, exist_ok=True)
    json.dump(label_options, open(f"{output_folder}/{dataset_name}-{model_name}-{pretrained}-idx.json", "w"))
    return label_options

