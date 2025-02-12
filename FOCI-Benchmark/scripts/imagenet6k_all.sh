##
#python run_ic_bench.py \
#--model=Salesforce/blip2-flan-t5-xl \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=Salesforce/blip2-flan-t5-xl \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=Salesforce/blip2-flan-t5-xl \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=Salesforce/blip2-flan-t5-xl \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

##
#python run_ic_bench.py \
#--model=adept/fuyu-8b \
#--batchsize=2 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=adept/fuyu-8b \
#--batchsize=2 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=adept/fuyu-8b \
#--batchsize=2 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=adept/fuyu-8b \
--batchsize=2 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/cache1/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \


##
#python run_ic_bench.py \
#--model=HuggingFaceM4/idefics2-8b \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=HuggingFaceM4/idefics2-8b \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=HuggingFaceM4/idefics2-8b \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=HuggingFaceM4/idefics2-8b \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/cache1/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \


##
#python run_ic_bench.py \
#--model=HuggingFaceM4/idefics-9b-instruct \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=HuggingFaceM4/idefics-9b-instruct \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=HuggingFaceM4/idefics-9b-instruct \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=HuggingFaceM4/idefics-9b-instruct \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \


##
#python run_ic_bench.py \
#--model=Salesforce/instructblip-flan-t5-xl \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=Salesforce/instructblip-flan-t5-xl \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=Salesforce/instructblip-flan-t5-xl \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=Salesforce/instructblip-flan-t5-xl \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \


##
#python run_ic_bench.py \
#--model=Salesforce/instructblip-vicuna-7b \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=Salesforce/instructblip-vicuna-7b \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=Salesforce/instructblip-vicuna-7b \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=Salesforce/instructblip-vicuna-7b \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \


##
#python run_ic_bench.py \
#--model=internlm/internlm-xcomposer2-vl-7b \
#--batchsize=2 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=internlm/internlm-xcomposer2-vl-7b \
#--batchsize=2 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=internlm/internlm-xcomposer2-vl-7b \
#--batchsize=2 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=internlm/internlm-xcomposer2-vl-7b \
--batchsize=2 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/cache1/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \


##
#python run_ic_bench.py \
#--model=llava-hf/llava-1.5-7b-hf \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=llava-hf/llava-1.5-7b-hf \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=llava-hf/llava-1.5-7b-hf \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



##
#python run_ic_bench.py \
#--model=llava-hf/llava-v1.6-mistral-7b-hf \
#--batchsize=1 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=llava-hf/llava-v1.6-mistral-7b-hf \
#--batchsize=1 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=llava-hf/llava-v1.6-mistral-7b-hf \
#--batchsize=1 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/cache1/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=llava-hf/llava-v1.6-mistral-7b-hf \
--batchsize=1 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/cache1/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



##
#python run_ic_bench.py \
#--model=Qwen/Qwen-VL-Chat \
#--batchsize=4 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=Qwen/Qwen-VL-Chat \
#--batchsize=4 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=Qwen/Qwen-VL-Chat \
#--batchsize=4 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=Qwen/Qwen-VL-Chat \
--batchsize=4 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



##
#python run_ic_bench.py \
#--model=mtgv/MobileVLM_V2-1.7B \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=mtgv/MobileVLM_V2-1.7B \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=mtgv/MobileVLM_V2-1.7B \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=mtgv/MobileVLM_V2-1.7B \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



##
#python run_ic_bench.py \
#--model=mtgv/MobileVLM_V2-3B \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=mtgv/MobileVLM_V2-3B \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=mtgv/MobileVLM_V2-3B \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=mtgv/MobileVLM_V2-3B \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



#
#python run_ic_bench.py \
#--model=mtgv/MobileVLM_V2-7B \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=mtgv/MobileVLM_V2-7B \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=mtgv/MobileVLM_V2-7B \
#--batchsize=8 \
#--workers=4 \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=mtgv/MobileVLM_V2-7B \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



##
#python run_ic_bench.py \
#--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
#--batchsize=8 \
#--workers=4 \
#--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_19_2024_15_34_08/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
#--prompt_query='Which of these animals is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-animal" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
#--batchsize=8 \
#--workers=4 \
#--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_19_2024_15_34_08/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
#--prompt_query='Which of these plants is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-plant" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \
#
#python run_ic_bench.py \
#--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
#--batchsize=8 \
#--workers=4 \
#--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_19_2024_15_34_08/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
#--prompt_query='Which of these foods is shown in the image?' \
#--model_cache_dir=/media/gregor/DATA/hf_cache \
#--dataset="imagenet-6k-food" \
#--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_19_2024_15_34_08/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



#
python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_45/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these animals is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-animal" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_45/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these plants is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-plant" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_45/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these foods is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-food" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_45/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \


#
python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_19_2024_15_35_45/checkpoints/0-5462.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these animals is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-animal" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_19_2024_15_35_45/checkpoints/0-5462.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these plants is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-plant" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_19_2024_15_35_45/checkpoints/0-5462.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these foods is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-food" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_19_2024_15_35_45/checkpoints/0-5462.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \


#
python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_40/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these animals is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-animal" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_40/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these plants is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-plant" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_40/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these foods is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-food" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_224.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_40/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



#
python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_336.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_53/checkpoints/0-3657.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these animals is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-animal" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_336.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_53/checkpoints/0-3657.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these plants is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-plant" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_336.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_53/checkpoints/0-3657.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these foods is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-food" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_large_patch14_clip_336.openai \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_38_53/checkpoints/0-3657.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \



#
python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_so400m_patch14_siglip_224 \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_44_13/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these animals is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-animal" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_so400m_patch14_siglip_224 \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_44_13/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these plants is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-plant" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_so400m_patch14_siglip_224 \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_44_13/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these foods is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-food" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \

python run_ic_bench.py \
--model=stabilityai/stablelm-2-zephyr-1_6b###vit_so400m_patch14_siglip_224 \
--batchsize=8 \
--workers=4 \
--model_revision=/media/gregor/DATA/projects/wuerzburg/lvlm-early-eval/checkpoints/instruct/03_20_2024_07_44_13/checkpoints/0-4877.ckpt/checkpoint/mp_rank_00_model_states.pt \
--prompt_query='Which of these choices is shown in the image?' \
--model_cache_dir=/media/gregor/DATA/hf_cache \
--dataset="imagenet-6k-artifact" \
--image_root=/media/gregor/cache1/icbench/imagenet6k \
