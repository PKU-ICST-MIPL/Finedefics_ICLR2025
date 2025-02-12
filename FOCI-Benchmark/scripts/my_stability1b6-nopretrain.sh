#
python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these animals is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-animal" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these plants is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-plant" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these foods is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-food" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these cars is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=stanford_cars \
--image_root=/media/gregor/cache1/icbench/stanfordcars/stanford_cars \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these pets is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=oxford_pet \
--image_root=/media/gregor/cache1/icbench/oxfordpets/oxford-iiit-pet \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these aircrafts is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=fgvc_aircraft \
--image_root=/media/gregor/cache1/icbench/fgvcaircraft/fgvc-aircraft-2013b \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these flowers is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=flowers102 \
--image_root=/media/gregor/cache1/icbench/flowers102/flowers-102/ \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these dishes is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=food101 \
--image_root=/media/gregor/cache1/icbench/food101/food-101 \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=imagenet \
--image_root=/media/gregor/cache1/icbench/imagenet/val \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=imagenet-rendition \
--image_root=/media/gregor/cache1/icbench/imagenetr/imagenet-r/ \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=imagenet-adversarial \
--image_root=/media/gregor/cache1/icbench/imageneta/imagenet-a/ \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=imagenet-sketch \
--image_root=/media/gregor/cache1/icbench/imagenetsketch/sketch/ \


python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/05_10_2024_08_22_12/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--dataset=geode \
--image_root=/media/gregor/cache1/icbench/geode \



