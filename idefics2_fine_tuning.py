# -*- coding: utf-8 -*-
"""Idefics2 - Fine-tuning example
"""
# !pip install -q git+https://github.com/huggingface/transformers.git
# !pip install -q accelerate datasets peft bitsandbytes

"""# Loading the model and the dataset

We load the model from the Hugging Face hub. `idefics2-8b` has gone through instruction fine-tuning on a large mixture of multimodal datasets and as such is a strong starting-point to fine-tune on your own use-case. We will start from this checkpoint.

To accommodate the GPU poors, the default hyper-parameters in this tutorial are chosen so that the fine-tuning takes less than 32 GB of GPU memory. For instance, an V100 in Google Colab should be sufficient.

If you happen to have more ressources, you are encouraged to revisit some of these constraints, in particular:
- Using 4 bit quantization
- Lora fine-tuning
- Freezing the vision encoder
- Small batch size compensated with higher gradient accumulation degree
- Deactivate image splitting
- Using flash-attention
"""

import argparse
import os

import torch
from accelerate import Accelerator
from peft import LoraConfig
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, Trainer, TrainingArguments

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(args):

    model_name_str = args.model_name.split("/")[-1].lower()
    model_id = f"{model_name_str}-{args.dataset_name}-{args.training_option}"
    output_dir = os.path.join(args.output_dir, model_id)
    os.makedirs(output_dir, exist_ok=True)
    

    # Example:
    DEVICE = "cuda:0"
    USE_LORA = args.training_option == "lora"
    USE_QLORA = args.training_option == "qlora"
    print(f"Lora: {USE_LORA}, QLora: {USE_QLORA}")

    # Assuming this would be the last line of your main setup and training logic
    print("Setup complete, starting training...")

    processor = AutoProcessor.from_pretrained(args.model_name, do_image_splitting=False)

    # Three options for training, from the lowest precision training to the highest precision training:
    # - QLora
    # - Standard Lora
    # - Full fine-tuning
    if args.add_lora_where == "projection":
        target_modules = ".*(modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj).*$"
    elif args.add_lora_where == "projection,resampler":
        target_modules = (
            ".*(modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
        )
    elif args.add_lora_where == "text_model,projection,resampler":
        target_modules = ".*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$"
    else:
        raise ValueError(f"Unknown target modules: {args.add_lora_where}")
    print(f"Target modules: {target_modules}")

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.2,
            target_modules=target_modules,
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian",
        )
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
        print(f"Loading model from: {args.model_name}")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args.model_name,
            device_map=device_map,
            low_cpu_mem_usage=True,
            
            quantization_config=bnb_config if USE_QLORA else None,
        )
        
        model.add_adapter(lora_config)
        model.enable_adapters()

    else:
        model = Idefics2ForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
    print_trainable_parameters(model)

    with open(os.path.join(output_dir, "model.json"), "w") as f:
        f.write(str(model))

    from datasets import load_dataset

    dataset_name = args.dataset_name
    train_dataset = load_dataset(
        "csv", data_files=f"data/{dataset_name}.csv"
    )["train"]
    
    # Determine the training stage from the dataset name
    # 1) Stage I: pretrain
    # 2) Stage II: finetune
    if "pretrain" in dataset_name.lower():
        model.training_stage = "pretrain"
    elif "finetune" in dataset_name.lower():
        model.training_stage = "finetune"
    else:
        raise ValueError

    """Let's look at an example. Each sample has two images, a question, and an answer."""

    print(train_dataset)
    print(train_dataset[0]["image_path"])

    """# Training loop

    We first define the data collator which takes list of samples and return input tensors fed to the model. There are 4 tensors types we are interested:
    - `input_ids`: these are the input indices fed to the language model
    - `attention_mask`: the attention mask for the `input_ids` in the language model
    - `pixel_values`: the (pre-processed) pixel values that encode the image(s). Idefics2 treats images in their native resolution (up to 980) and their native aspect ratio
    - `pixel_attention_mask`: when multiple image(s) are packed into the same sample (or in the batch), attention masks for the images are necessary because of these images can have different sizes and aspect ratio. This masking ensures that the vision encoder properly forwards the images.

    """
    class MyDataCollator:
        def __init__(self, processor):
            self.processor = processor
            self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
                processor.tokenizer.additional_special_tokens.index("<image>")
            ]

        def __call__(self, examples):
            texts = []
            images = []
            attributes = []
            neg_attributes = []
            categories = []
            neg_categories = []
            for example in examples:
                question = example["question"]
                answer = example["answer"]
                if example["type"] == "multi-image":
                    user_content = [
                        {"type": "image"},
                        {"type": "image"},
                    ]
                    ex_images = [example["image_path_1"], example["image_path_2"]]
                else:
                    user_content = [{"type": "image"}]
                    ex_images = [example["image_path"]]

                if question:
                    user_content.append({"type": "text", "text": question})
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
                text = processor.apply_chat_template(messages, add_generation_prompt=False)
                
                if "base" in args.model_name:  # hack to remove the end of utterance token
                    text = text.replace("<end_of_utterance>", "")
                texts.append(text.strip())
                ex_images = [Image.open(os.path.join(args.image_path, img_path)).convert("RGB") for img_path in ex_images]
                images.append(ex_images)
                if example["attribute"] is not None:
                    attributes.append(example["attribute"])
                if example["neg_attribute"] is not None:
                    neg_attributes.extend(example["neg_attribute"].split("\t\n"))
                if example["category"] is not None:
                    categories.append(example["category"])
                if example["neg_category"] is not None:
                    neg_categories.extend(example["neg_category"].split("\t\n"))

            if len(attributes) == 0:
                attributes = None
            if len(neg_attributes) == 0:
                neg_attributes = None
            if len(categories) == 0:
                categories = None
            if len(neg_categories) == 0:
                neg_categories = None
                
                
            batch, att_tokens, neg_att_tokens, cat_tokens, neg_cat_tokens = processor(text=texts, images=images, return_tensors="pt", padding=True, attribute=attributes, category=categories, neg_attribute=neg_attributes, neg_category=neg_categories)
            labels = batch["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
            batch["labels"] = labels

            if cat_tokens is not None:
                if att_tokens is not None:
                    batch.update({"att_tokens": att_tokens, "cat_tokens": cat_tokens} if neg_att_tokens is None else {"att_tokens": att_tokens, "cat_tokens": cat_tokens, "neg_att_tokens": neg_att_tokens, "neg_cat_tokens": neg_cat_tokens}) #, "eos_token_id": torch.tensor([[processor.tokenizer.eos_token_id]])})
                else:
                    #print(cat_tokens)
                    batch.update({"cat_tokens": cat_tokens} if neg_cat_tokens is None else {"cat_tokens": cat_tokens, "neg_cat_tokens": neg_cat_tokens})#, "eos_token_id": torch.tensor([[processor.tokenizer.eos_token_id]])})
     
            return batch

    data_collator = MyDataCollator(processor)

    """We will use HuggingFace Trainer."""

    num_devices = torch.cuda.device_count() * args.num_nodes
    gradient_accumulation_steps = max(1, args.batch_size // (args.batch_size_per_device * num_devices))

    model_name_str = args.model_name.split("/")[-1].lower() 
    model_id = f"{model_name_str}-{dataset_name}-{args.training_option}"
    output_dir = os.path.join(args.output_dir, model_id)
    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=2,
        log_level="info",
        output_dir=output_dir,
        overwrite_output_dir=args.resume_from_checkpoint is None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    resume_from_checkpoint = None
    if args.resume_from_checkpoint is not None:
        from transformers.trainer_utils import get_last_checkpoint

        print(f"Searching for last checkpoint in {output_dir}")
        resume_from_checkpoint = get_last_checkpoint(output_dir)
        if resume_from_checkpoint is not None:
            print(f"Resuming training from {resume_from_checkpoint}")
            model = Idefics2ForConditionalGeneration.from_pretrained(
                resume_from_checkpoint,
                device_map=device_map,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config if USE_QLORA else None,
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    """# Training and pushing to the hub

    We have all the core building blocks now, so we fine-tune the model!

    The training can take a few minutes depending on the hardware you use.
    """

    trainer.train(resume_from_checkpoint=resume_from_checkpoint) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is used for fine-tuning the IDEFICS2 model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="pretrained_weights/idefics2-8b",
        help="Specify the name of the model to be trained.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/train/",
        help="Specify the source of the training data.",
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Specify the path to the checkpoint from which the training should be resumed.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pretrain",
        help="Specify the name of the dataset to be used for fine-tuning the model.",
    )
    parser.add_argument(
        "--training_option",
        type=str,
        default="qlora",
        choices=["qlora", "lora", "full"],
        help="Choose the training option: qlora for QLora (lowest precision training), lora for Standard Lora (medium precision training), full for Full fine-tuning (highest precision training)",
    )
    parser.add_argument(
        "--add_lora_where",
        type=str,
        default="text_model,projection,resampler",
        choices=["projection", "projection,resampler", "text_model,projection,resampler"],
        help="Choose the target modules for Lora/QLora adapter.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Specify the number of epochs for training the model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Specify the batch size for training the model.")
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=8,
        help="Specify the batch size per device for training the model.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Specify the learning rate for training the model."
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="linear", help="Specify the learning rate scheduler type."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=60, help="Specify the number of warmup steps for training the model."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Specify the weight decay for training the model."
    )
    parser.add_argument("--lora_r", type=int, default=8, help="Specify the r value for Lora.")
    parser.add_argument("--lora_alpha", type=int, default=8, help="Specify the alpha value for Lora.")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Specify the directory where the model checkpoints will be saved.",
    )
    parser.add_argument("--save_steps", type=int, default=60, help="Specify the number of steps to save the model.") 
    args = parser.parse_args()
    main(args)