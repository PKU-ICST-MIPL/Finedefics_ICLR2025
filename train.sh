#!/bin/bash

set -e

### Stage I: Attribute Augmented Contrastive Learning

# Pretraining
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 6023 --nproc_per_node 8 idefics2_fine_tuning.py --model_name pretrained_weights/idefics2-8b --image_path data --batch_size_per_device 2 --batch_size 64 --training_option qlora --epochs 1 --dataset_name pretrain --add_lora_where text_model,projection,resampler --warmup_steps 60 
# Merge QLoRA Weights
python qlora_to_merge.py --base_model pretrained_weights/idefics2-8b --qlora_model checkpoints/idefics2-8b-pretrain-qlora/checkpoint-602 --new_model checkpoints/idefics2-8b-pretrain-qlora-merge

### Stage II: Classification-Centered Instruction Tuning

# Finetuning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 6024 --nproc_per_node 8 idefics2_fine_tuning.py --model_name checkpoints/idefics2-8b-pretrain-qlora-merge --image_path data --batch_size_per_device 8 --batch_size 128 --training_option qlora --epochs 1 --dataset_name finetune --add_lora_where text_model,projection,resampler --warmup_steps 60 
# Merge QLoRA Weights
python qlora_to_merge.py --base_model checkpoints/idefics2-8b-pretrain-qlora-merge --qlora_model checkpoints/idefics2-8b-pretrain-qlora-merge-finetune-qlora/checkpoint-602 --new_model checkpoints/idefics2-8b-pretrain-qlora-merge-finetune-qlora-merge

