
python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these pets is shown in the image?' \
--dataset=oxford_pet \
--image_root=/media/gregor/cache1/icbench/oxfordpets/oxford-iiit-pet \
#
python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these cars is shown in the image?' \
--dataset=stanford_cars \
--image_root=/media/gregor/cache1/icbench/stanfordcars/stanford_cars \

python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these aircrafts is shown in the image?' \
--dataset=fgvc_aircraft \
--image_root=/media/gregor/cache1/icbench/fgvcaircraft/fgvc-aircraft-2013b \

python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these flowers is shown in the image?' \
--dataset=flowers102 \
--image_root=/media/gregor/cache1/icbench/flowers102/flowers-102/ \

python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these dishes is shown in the image?' \
--dataset=food101 \
--image_root=/media/gregor/cache1/icbench/food101/food-101 \


python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--dataset=imagenet \
--image_root=/media/gregor/cache1/icbench/imagenet/val \

python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--dataset=imagenet-rendition \
--image_root=/media/gregor/cache1/icbench/imagenetr/imagenet-r/ \

python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--dataset=imagenet-adversarial \
--image_root=/media/gregor/cache1/icbench/imageneta/imagenet-a/ \

python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--dataset=imagenet-sketch \
--image_root=/media/gregor/cache1/icbench/imagenetsketch/sketch/ \


python run_ic_bench.py \
--model=llava-hf/llava-1.5-7b-hf \
--batchsize=8 \
--workers=4 \
--prompt_query='Which of these choices is shown in the image?' \
--dataset=geode \
--image_root=/media/gregor/cache1/icbench/geode \

