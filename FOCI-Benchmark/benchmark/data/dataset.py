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


class ICDataset(Dataset):
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

class BabelImageNetICDataset(Dataset):
    def __init__(self, args):
        dataset = args.dataset
        language = dataset.split('-')[-1].upper()
        loader = DATASET_TO_LOADER["imagenet"]
        label2paths = loader(args.image_root)
        label_idx_options = generate_options_idx(label2paths, image_root=args.image_root, output_folder=args.data_output_folder, dataset_name="imagenet")

        bin = json.load(open(f"{args.data_output_folder}/babel_imagenet-298.json"))
        class_idxs, class_labels = bin[language]
        class_idxs_set = set(class_idxs)
        idx2label = {idx:label for idx, label in zip(class_idxs, class_labels)}
        self.examples = []
        choices = 4
        max_per_label = args.max_examples
        for label_idx, examples in label_idx_options.items():
            label_idx = int(label_idx)
            if label_idx not in class_idxs_set:
                continue
            for example in examples[:max_per_label]:
                options_idxs = example["options"]
                gt_idx = options_idxs[-1] # last example is always ground truth
                other_choices = []
                for choice in reversed(options_idxs[:-1]):
                    if len(other_choices) == choices-1:
                        break
                    if choice in class_idxs_set:
                        other_choices.append(choice)
                labels = [idx2label[choice] for choice in [gt_idx]+other_choices]
                example["options"] = labels
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

def model_template(model):
    if "mistral" in model:
        template ="[INST] <image>\n{} [/INST]"
    elif "llava" in model or "Mobile" in model:
        template = "USER: <image>\n{}\nASSISTANT:"
    elif "idefics" in model:
        template = "User: {}"
    elif "Qwen" in model and "Chat" in model:
        template = "<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n"
    elif "stability" in model:
        template = "<|user|><filename>\n{}<|endoftext|>\n<|assistant|>\n"
    elif "fuyu" in model:
        template = "{}\n"
    elif "Phi-3-vision" in model:
        template = "<|user|>\n<|image_1|>\n{}<|end|>\n<|assistant|>\n"
    elif "Yi" in model:
        template = "### Human: <image_placeholder>\n{}\n### Assistant:"
    else:
        template = "{}"

    return template


def prepare_prompt(model, options, task, **kwargs):
    if task == "mc":
        return prepare_prompt_mc(model, options, **kwargs)
    elif task == "yn":
        return prepare_prompt_yn(model, options, **kwargs)
    else:
        raise ValueError("Unknown task: {}".format(task))
def prepare_prompt_mc(model, options,
                   prompt_query="Which of these choices is shown in the image?",
                   prompt_options="\nChoices:\n",  #Choices:
                   prompt_end="\nAnswer with the letter from the given choices directly.",
                   choice_enumeration="ABCD"):  #\nAnswer with the letter from the given choices directly.

    shuffle_idx = np.random.permutation(len(options)).tolist()
    gold_idx = shuffle_idx.index(0)
    gold_label = choice_enumeration[gold_idx]
    formatted_options = "\n".join([f"{choice_enumeration[i]}. {options[j]}" for i, j in enumerate(shuffle_idx)])
    label2option = {choice_enumeration[i]: options[j] for i, j in enumerate(shuffle_idx)}
    formatted_prompt = f"{prompt_query}{prompt_options}{formatted_options}{prompt_end}"
    formatted_prompt = model_template(model).format(formatted_prompt)
    return formatted_prompt, gold_label, label2option

def prepare_prompt_yn(model, options,
                   prompt_query="Does the image show a/an {}? Answer with yes or no.",
                   prompt_options="\nChoices: ",  #Choices:
                   prompt_end="\nAnswer with the letter from the given choices directly.",
                   choice_enumeration="ABCD"):  #\nAnswer with the letter from the given choices directly.

    if random.random() < 0.5:
        formatted_prompt = prompt_query.format(options[0]).replace("a/an", "an" if options[0][0] in "aeiouAEIOU" else "a")
        gold_label = "yes"
    else:
        formatted_prompt = prompt_query.format(options[-1]).replace("a/an", "an" if options[0][0] in "aeiouAEIOU" else "a")
        gold_label = "no"
    formatted_prompt = model_template(model).format(formatted_prompt)
    label2option = {"true": options[0], "false": options[-1]}
    return formatted_prompt, gold_label, label2option

# class MCCollator:
#     def __init__(self, args):
#         model = args.model
#         image_root = args.image_root
#         self.model = model
#         self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
#         if not self.processor.tokenizer.pad_token:
#             self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
#         if not "t5" in model or not "t0" in model:
#             self.processor.tokenizer.padding_side = "left"
#         self.image_root = image_root
#         self.prompt_query = args.prompt_query #Which of these choices is shown in the image?
#
#     def __call__(self, batch):
#         options = [b["options"] for b in batch]
#         prompts_labels_mapping = [prepare_prompt(self.model, option, prompt_query=self.prompt_query) for option in options]
#         prompts, labels, mapping = list(zip(*prompts_labels_mapping))
#         image_files = [b["image"] for b in batch]
#         images = [Image.open(os.path.join(self.image_root, b["image"])) for b in batch]
#
#         inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
#         return inputs, labels, mapping, image_files
#



