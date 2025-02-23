<!-- PROJECT LOGO -->

<p align="center">
  <h1 align="center">Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language Models</h1>
  <p align="center">
    <a href="http://39.108.48.32/mipl/news/news.php?id=EGhehulingxiao"><strong>Hulingxiao He</strong></a>
    ·
    <a href="http://39.108.48.32/mipl/news/news.php?id=EGligeng"><strong>Geng Li</strong></a>
    ·
    <a href="http://39.108.48.32/mipl/news/news.php?id=EGgengzijun"><strong>Zijun Geng</strong></a>
    ·
    <a href="https://github.com/xujinglin"><strong>Jinglin Xu</strong></a>
    ·
    <a href="http://39.108.48.32/mipl/yuxinpeng/"><strong>Yuxin Peng</strong></a>
  </p>
  <h2 align="center">ICLR 2025</h2>
  <h3 align="center"><a href="https://openreview.net/forum?id=p3NKpom1VL">OpenReview</a> | <a href="https://arxiv.org/abs/2501.15140">Paper</a> | <a href="https://huggingface.co/StevenHH2000/Finedefics">Model</a> </h3>
<div align="center"></div>
<p align="center">
  <p>
  <strong>TL;DR</strong>: We revisit three quintessential capabilities of MLLMs for FGVR, including object information extraction, category knowledge reserve, object-category alignment, and position of the root cause as <strong> a misalignment problem</strong>. To address this issue, we present <strong> Finedefics</strong>, an MLLM that enhances the model's FGVR capability by incorporating informative attribute descriptions of objects into the training phase. 
  </p>
  <a href="">
    <img src="figures/pipeline.png" alt="Logo" width="100%">
  </a>
<br>


## 📣 News
- [02/12/2025] We release the model <a href="https://huggingface.co/StevenHH2000/Finedefics"><strong>Finedefics</strong></a> and evaluation code.
- [01/23/2025] Our work is accepted to <a href="https://iclr.cc/Conferences/2025"><strong>ICLR 2025</strong></a> 🌼! Code is coming soon. See you in Singapore this April!

## ⚗ Training

The training code is coming soon. Stay tuned!


## 📋 Evaluation
We use [FOCI-Benchmark](https://github.com/gregor-ge/FOCI-Benchmark) to evaluate our model.

#### 1. Preparing the Data
Before starting, we can download & prepare the evaluation datasets we want to use following a guide [here](FOCI-Benchmark/DATA.md).


#### 2. Preparing the Environment
Requirements can be found in [requirements.txt](FOCI-Benchmark/requirements.txt). We recommend using Python ≥ 3.9 and PyTorch ≥ 2.2.1.


#### 3. Running the Benchmark


An example of evaluating on `dog-120` dataset is:
```
python run_ic_bench.py --model=/path/to/model --dataset=dog-120 --prompt_query='Which of these dogs is shown in the image?' --image_root=/path/to/dog-120 --batchsize=4
```

Note: Avaliable datasets are `dog-120, bird-200, fgvc_aircraft, flowers102, oxford_pet, stanford_cars, imagenet-rendition, imagenet-adversarial, imagenet-sketch`.

See [scripts](FOCI-Benchmark/scripts/Finedefics.sh) for examples of evaluating Finedefics on all benchmark datasets.


#### 4. Testing New Models

Our code is trivial to extend to new models, especially if they use HuggingFace:

- Implement the model based on the reference [HfModel](FOCI-Benchmark/benchmark/model/model.py) or the other  implemented models.
- Update [model_template()](FOCI-Benchmark/benchmark/data/dataset.py) to provide the model instruction template.
- Update [load_model()](FOCI-Benchmark/benchmark/model/model.py) to load the model based on the name.

#### 5. Testing on New Datasets

Our code is also trivial to extend to new image classification datasets:

- Implement a loader function that creates a dictionary mapping labels to (relative) image paths and add it to [DATASET_TO_LOADER](FOCI-Benchmark/benchmark/data/dataset.py).
- When running the benchmark for the first time, we use CLIP to find difficult multiple-choice options and store them in [data](FOCI-Benchmark/data) for subsequent runs.

## 🚩 Acknowledgments
Our code references [FineR](https://github.com/OatmealLiu/FineR?tab=readme-ov-file#%EF%B8%8F-full-pipeline), [FOCI-Benchmark](https://github.com/gregor-ge/FOCI-Benchmark), [HACL](https://github.com/X-PLUG/mPLUG-HalOwl/tree/main/hacl). Many thanks to the authors.

## 🗻 Citation
Should you find our paper valuable to your work, we would greatly appreciate it if you could cite it:
```bibtex
@inproceedings{
    he2025analyzing,
    title={Analyzing and Boosting the Power of Fine-Grained Visual Recognition for Multi-modal Large Language Models},
    author={Hulingxiao He and Geng Li and Zijun Geng and Jinglin Xu and Yuxin Peng},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=p3NKpom1VL}
}
```