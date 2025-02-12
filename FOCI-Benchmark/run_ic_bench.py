import argparse
from benchmark.evaluate import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--image_root')
    parser.add_argument('--max_examples', type=int, default=9999)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--results_output_folder', type=str, default='./results')
    parser.add_argument('--data_output_folder', type=str, default='./data')
    parser.add_argument('--model_revision', type=str, default='main', help="HuggingFace model revision")
    parser.add_argument('--load_quantized', default=False)
    parser.add_argument('--model_cache_dir', default=None)
    parser.add_argument('--prompt_query', type=str, default="")
    parser.add_argument('--choice_enumeration', type=str, default='ABCD')
    parser.add_argument('--task', type=str, default='mc')

    args = parser.parse_args()
    main(args)