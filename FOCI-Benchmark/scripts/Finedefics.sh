python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=flowers102 
--prompt_query='Which of these flowers is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/flower102 
--batchsize=4

python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=stanford_cars 
--prompt_query='Which of these cars is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/stanford-cars 
--batchsize=2

python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=fgvc_aircraft 
--prompt_query='Which of these aircrafts is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/fgvc_aircraft 
--batchsize=2

python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=bird_200 
--prompt_query='Which of these birds is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/bird_200 
--batchsize=2

python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=dogs_120 
--prompt_query='Which of these dogs is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/dogs_120 
--batchsize=2

python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=oxford_pet 
--prompt_query='Which of these pets is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/oxford-pet 
--batchsize=2

python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=imagenet-rendition 
--prompt_query='Which of these choices is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/imagenet-r 
--batchsize=4

python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=imagenet-adversarial 
--prompt_query='Which of these choices is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/imagenet-a 
--batchsize=4

python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=imagenet-sketch 
--prompt_query='Which of these choices is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/ImageNet-Sketch/sketch 
--batchsize=4

python run_ic_bench.py 
--model=/data/hehulingxiao/code/FineIdefics2/checkpoints/idefics2-8b-pretrain_siglip-qlora-lr0.0002-merge-finetune-qlora-lr0.0002-merge 
--dataset=geode 
--prompt_query='Which of these choices is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/geode 
--batchsize=4

