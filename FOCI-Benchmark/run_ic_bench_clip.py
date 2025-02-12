import argparse
from benchmark.clip_benchmark.evaluate import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--from_pretrained')
    parser.add_argument('--from_pretrained_dataset')
    parser.add_argument('--image_root')
    parser.add_argument('--max_examples', type=int, default=9999)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--results_output_folder', type=str, default='./results_clip')
    parser.add_argument('--template', type=str, default="an image of a {}")
    parser.add_argument('--data_output_folder', type=str, default='./data')
    parser.add_argument('--model_cache_dir', default=None)

    args = parser.parse_args()
    main(args)