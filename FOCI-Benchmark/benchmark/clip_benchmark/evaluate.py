import json
import os
import torch
import torch.nn.functional as F
import torchvision.datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.clip_benchmark.dataset import ImageDataset, TextDataset
from benchmark.data.loaders import DATASET_TO_LOADER
from benchmark.data.options_generation import generate_options
from benchmark.clip_benchmark.model import OpenCLIPModel

def compute_accuracy(image_embeddings, text_embeddings, target, label_subset, num_prompts=1, ):
    # Prompt ensembles are averaged for the final prompt embedding
    if num_prompts > 1:
        text_embeddings = text_embeddings.view(len(text_embeddings)//num_prompts, num_prompts, -1)
        text_embeddings = torch.mean(text_embeddings, dim=1)
        text_embeddings = F.normalize(text_embeddings, dim=1)

    scores = image_embeddings @ text_embeddings.t()
    scores = torch.softmax(scores, dim=-1)
    confidence, prediction = scores.max(dim=1)
    confidence = confidence * 100
    target = torch.tensor(target).to(scores.device)
    accuracy = (target == prediction).sum().item() / target.size(0)

    label_subset = torch.tensor(label_subset).to(scores.device)
    scores4 = torch.gather(scores, 1, label_subset)
    confidence4, prediction4 = scores4.max(dim=1)
    prediction4 = torch.gather(label_subset, 1, prediction4.unsqueeze(1)).squeeze()
    confidence4 = confidence4 * 100
    accuracy4 = (target == prediction4).sum().item() / target.size(0)

    return accuracy, prediction.cpu().tolist(), confidence.cpu().tolist(), accuracy4, prediction4.cpu().tolist(), confidence4.cpu().tolist()


def compute_image_embeddings(image_dataset, model, args):
    dataloader = DataLoader(image_dataset, batch_size=args.batchsize, num_workers=args.workers)
    all_embeddings = None
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, desc="Encoding images"):
            batch = batch.to("cuda")
            embeddings = model.encode_images(batch)
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    return all_embeddings


def compute_text_embeddings(text_dataset, collate, model, args):
    dataloader = DataLoader(text_dataset, batch_size=args.batchsize, num_workers=args.workers, collate_fn=collate)
    all_embeddings = None
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, desc="Encoding text"):
            batch = {k: v.to("cuda") for k,v in batch.items()}
            embeddings = model.encode_text(**batch)
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.cat((all_embeddings, embeddings), dim=0)
    return all_embeddings

def main(args):
    model = OpenCLIPModel(args.from_pretrained, args.from_pretrained_dataset, args.model_cache_dir)
    model.eval()
    model.to("cuda")
    text_collate = model.tokenize

    dataset = args.dataset
    loader = DATASET_TO_LOADER[dataset]
    label2paths = loader(args.image_root)
    label_options = generate_options(label2paths, image_root=args.image_root, output_folder=args.data_output_folder,
                                     dataset_name=dataset)
    labels = list(label_options.keys())

    images, targets, label_subset = [], [], []
    for label, options in label_options.items():
        for option in options:
            images.append(option["image"])
            targets.append(labels.index(option["options"][0]))
            label_subset.append([labels.index(o) for o in option["options"]])

    image_dataset = ImageDataset(args.image_root, images, model.transform)

    template = args.template
    prompts = [template.format(l) for l in labels]
    text_dataset = TextDataset(prompts)

    image_embeddings = compute_image_embeddings(image_dataset, model, args)
    text_embeddings = compute_text_embeddings(text_dataset, text_collate, model, args)

    acc, predictions, confidence, acc4, prediction4, confidence4 = compute_accuracy(image_embeddings, text_embeddings, targets, label_subset, num_prompts=1)
    print(f"{args.dataset} {args.from_pretrained} | acc: {acc:.4f} / {acc4:.4f}")
    results = []
    for p, c, i, t, p4, c4 in zip(predictions, confidence, images, targets, prediction4, confidence4):
        results.append(dict(image=i, prediction=labels[p], prediction4=labels[p4], target=labels[t]))

    output = {"metrics": dict(acc=acc, acc4=acc4), "predictions": results, "config": vars(args)}

    os.makedirs(args.results_output_folder, exist_ok=True)
    json.dump(output,
              open(f"{args.results_output_folder}/clip--{args.dataset}--{args.from_pretrained.replace('/', '_')}-{args.from_pretrained_dataset}.json", "w"), indent=4)