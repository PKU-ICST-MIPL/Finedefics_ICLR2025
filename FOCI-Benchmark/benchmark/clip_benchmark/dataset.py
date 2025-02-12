import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
from PIL import Image
from benchmark.data.loaders import DATASET_TO_LOADER
from benchmark.data.options_generation import generate_options, generate_options_idx


class ICCLIPDataset(Dataset):
    def __init__(self, args):
        dataset = args.dataset
        loader = DATASET_TO_LOADER[dataset]
        label2paths = loader(args.image_root)
        label_options = generate_options(label2paths, image_root=args.image_root, output_folder=args.data_output_folder, dataset_name=dataset)
        self.examples = []
        max_per_label = args.max_examples
        for label, options in label_options.items():
            for example in options[:max_per_label]:
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

class ImageDataset(Dataset):
    def __init__(self, root, images, transform):
        self.root = root
        self.transform = transform
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        path = os.path.join(str(self.root), str(self.images[item]))
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img

class TextDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __getitem__(self, item):
        return self.prompts[item]

    def __len__(self):
        return len(self.prompts)