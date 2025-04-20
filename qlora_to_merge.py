'''
Author: StevenHH2000 hlxhe2000@hotmail.com
Date: 2025-04-14 11:49:26
LastEditors: StevenHH2000 hlxhe2000@hotmail.com
LastEditTime: 2025-04-18 02:09:18
FilePath: /code/ICLR2025/training/qlora_to_merge.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Merge the model
from transformers import Idefics2ForConditionalGeneration, LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel

import argparse

def main(args):
    base_model = Idefics2ForConditionalGeneration.from_pretrained(args.base_model)
    p_model = PeftModel.from_pretrained(base_model, model_id=args.qlora_model) # / 'checkpoint-602') #630')
    merge_model = p_model.merge_and_unload()
    merge_model.save_pretrained(args.new_model) # / 'checkpoint-602') #630')
    processor = AutoProcessor.from_pretrained(
        args.base_model                                                                                                                                         , 
        do_image_splitting=False)
    processor.save_pretrained(args.new_model) # / 'checkpoint-602') #630')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script is used for fine-tuning the IDEFICS2 model.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="pretrained_weights/idefics2-8b/",
        help="Specify the name of the model to be merged.",
    )
    parser.add_argument(
        "--qlora_model",
        type=str,
        default="checkpoints/idefics2-8b-pretrain-qlora-lr0.0002/checkpoint-602",
        help="Specify the name of the qlora weights.",
    )
    parser.add_argument(
        "--new_model",
        type=str,
        default="checkpoints/idefics2-8b-pretrain-qlora-lr0.0002-merge",
        help="Specify the name of the model after merging.",
    )
    
    args = parser.parse_args()
    main(args)