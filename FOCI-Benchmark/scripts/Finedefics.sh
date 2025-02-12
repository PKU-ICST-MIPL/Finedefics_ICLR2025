# Dog-120
python run_ic_bench.py 
--model=/path/to/model
--dataset=dogs_120 
--prompt_query='Which of these dogs is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/dogs_120 
--batchsize=2

# Bird-200
python run_ic_bench.py 
--model=/path/to/model
--dataset=bird_200 
--prompt_query='Which of these birds is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/bird_200 
--batchsize=2

# Aircraft-102
python run_ic_bench.py 
--model=/path/to/model
--dataset=fgvc_aircraft 
--prompt_query='Which of these aircrafts is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/fgvc_aircraft 
--batchsize=2

# Flower-102
python run_ic_bench.py 
--model=/path/to/model
--dataset=flowers102 
--prompt_query='Which of these flowers is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/flower102 
--batchsize=4

# Pet-37
python run_ic_bench.py 
--model=/path/to/model 
--dataset=oxford_pet 
--prompt_query='Which of these pets is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/oxford-pet 
--batchsize=2

# Car-196
python run_ic_bench.py 
--model=/path/to/model
--dataset=stanford_cars 
--prompt_query='Which of these cars is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/stanford-cars 
--batchsize=2

# IN-rendition
python run_ic_bench.py 
--model=/path/to/model
--dataset=imagenet-rendition 
--prompt_query='Which of these choices is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/imagenet-r 
--batchsize=4

# IN-adversarial
python run_ic_bench.py 
--model=/path/to/model
--dataset=imagenet-adversarial 
--prompt_query='Which of these choices is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/imagenet-a 
--batchsize=4

# IN-sketch
python run_ic_bench.py 
--model=/path/to/model
--dataset=imagenet-sketch 
--prompt_query='Which of these choices is shown in the image?' 
--image_root=/media/gregor/cache1/icbench/ImageNet-Sketch/sketch 
--batchsize=4

