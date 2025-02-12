import json
import os

from torch.utils.data import DataLoader

from benchmark.data.dataset import ICDataset, BabelImageNetICDataset  # , MCCollator
from benchmark.model.model import HFModel, load_model
from tqdm import tqdm

def parse_generated_prediction(predictions, choice_enumeration="ABCD"):
    parsed = []
    for prediction in predictions:
        match = False
        for letter in choice_enumeration:
            if prediction.lower().startswith(letter.lower()):
                parsed.append(letter)
                match = True
                break
        if not match:
            parsed.append(f"NA ### {prediction}")
    return parsed

def main(args):
    if "babel" in args.dataset:
        dataset = BabelImageNetICDataset(args)
    else:
        dataset = ICDataset(args)
    print(args.dataset, args.model)
    model = load_model(args)
    collate = model.collate #MCCollator(args)

    dataloader = DataLoader(dataset, collate_fn=collate, batch_size=args.batchsize, num_workers=args.workers)

    results = []
    correct = 0
    generation_failures = 0

    if args.task == "mc":
        for (inputs, labels, mapping, image_files) in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs[0].to("cuda")
            prediction = model.generate(inputs)
            prediction_parsed = parse_generated_prediction(prediction, args.choice_enumeration)

            for p, r, l, m, i in zip(prediction_parsed, prediction, labels, mapping, image_files):
                if p.startswith("NA"):
                    generation_failures += 1
                elif p == l:
                    correct += 1
                predicted = m.get(p, p)
                results.append({
                    "image": i,
                    "predicted": predicted,
                    "predicted_raw": r,
                    "correct": m[l],
                    "options": list(m.values())
                })
                
    elif args.task == "yn":
        for (inputs, labels, mapping, image_files) in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to("cuda")
            prediction = model.generate(inputs)

            for r, l, m, i in zip(prediction, labels, mapping, image_files):
                if r.lower().startswith(l):
                    correct += 1
                if not r.lower().startswith("yes") and not r.lower().startswith("no"):
                    generation_failures += 1
                results.append({
                    "image": i,
                    "predicted": r,
                    "predicted_raw": r,
                    "correct": l,
                    "options": list(m.values())
                })

    metrics = {
        "generation_failures": generation_failures/len(results),
        "accuracy": correct/len(results)
    }
    output = {"metrics": metrics, "predictions": results, "config": vars(args)}
    print(args.dataset, args.model)
    print(metrics)
    print("#################################")

    os.makedirs(os.path.join(args.results_output_folder, args.task), exist_ok=True)

    revision_filename = ""
    if args.model_revision != 'main':
        revision_filename = '-' + args.model_revision
    if "###" in args.model:
        revision_filename = '-' + args.model_revision.split("instruct/")[1].split("/checkpoints")[0]  #hard coded for my folder naming scheme


    json.dump(output,
              open(f"{args.results_output_folder}/{args.task}/{args.task+'--' if args.task != 'mc' else ''}{args.dataset}--{args.model.replace('/', '_')}{revision_filename}.json", "w"), indent=4)