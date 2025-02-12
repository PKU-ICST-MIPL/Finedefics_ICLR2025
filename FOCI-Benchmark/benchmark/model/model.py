import os

import torch
from PIL import Image
from torchvision.transforms import transforms, InterpolationMode
from transformers import AutoModelForCausalLM, AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig, \
    InstructBlipForConditionalGeneration, LlavaForConditionalGeneration, IdeficsForVisionText2Text, GenerationConfig, \
    AutoTokenizer, FuyuForCausalLM, AutoModelForVision2Seq
from transformers import AddedToken
# from transformers import LlavaNextForConditionalGeneration, PaliGemmaForConditionalGeneration
from benchmark.data.dataset import prepare_prompt
from benchmark.model.model_internlm_xcomposer2 import InternLMXComposer2ForCausalLM


def load_model(args):
    if "idefics2" in args.model or "Finedefics" in args.model:
        return Idefics2Model(args)
    elif "idefics" in args.model:
        return IdeficsModel(args)
    elif "Qwen" in args.model:
        return QwenVLModel(args)
    elif "Mobile" in args.model:
        from benchmark.model.model_mobilevlm import MobileVLM
        return MobileVLM(args)
    elif "###" in args.model:
        from benchmark.model.my_llava import LlavaModel
        return LlavaModel(args)
    elif "internlm" in args.model:
        return InternLMXComposerModel(args)
    # elif "Yi" in args.model:
    #     from benchmark.model.model_yivl import YiVL
    #     return YiVL(args)
    else:
        return HFModel(args)



class HFModel:
    def __init__(self, args):
        model = args.model
        load_quantized = args.load_quantized
        cache_dir = args.model_cache_dir
        revision = args.model_revision

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_quantized == "4bit",
            load_in_8bit=load_quantized == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )


        if "blip" in model and "instruct" in model:
            self.model = InstructBlipForConditionalGeneration.from_pretrained(model,
                                                              quantization_config=bnb_config if load_quantized else None,
                                                              torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                                                                              revision=revision)
        elif "blip" in model:
            self.model = Blip2ForConditionalGeneration.from_pretrained(model,
                                                              quantization_config=bnb_config if load_quantized else None,
                                                              torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                                                                              revision=revision)

        elif "llava-1.5" in model and not "1.6" in model:
            self.model = LlavaForConditionalGeneration.from_pretrained(model,
                                                              quantization_config=bnb_config if load_quantized else None,
                                                              torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                                                                              revision=revision)
        elif "llava-hf" in model and "1.6" in model:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model,
                                                              quantization_config=bnb_config if load_quantized else None,
                                                              torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                                                                              revision=revision, use_flash_attention_2=True)
        elif "fuyu" in model:
            self.model = FuyuForCausalLM.from_pretrained(model,
                                                              quantization_config=bnb_config if load_quantized else None,
                                                              torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                                                         low_cpu_mem_usage=True)
        elif "pali" in model:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                                                                    model,
                                                                    torch_dtype=torch.bfloat16,
                                                                    revision="bfloat16",
                                                                ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model,
                                                              quantization_config=bnb_config if load_quantized else None,
                                                              torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                                                              trust_remote_code=True,
                                                                              revision=revision)
        if not bnb_config.load_in_4bit and not bnb_config.load_in_8bit:
            self.model = self.model.to("cuda")

        self.model_name = model
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        # Bilp update
        if 'blip' in model:
            self.processor.num_query_tokens = self.model.config.num_query_tokens
            image_token = AddedToken("<image>", normalized=False, special=True)
            self.processor.tokenizer.add_tokens([image_token], special_tokens=True)

            self.model.resize_token_embeddings(len(self.processor.tokenizer), pad_to_multiple_of=64) # pad for efficient computation
            self.model.config.image_token_index = len(self.processor.tokenizer) - 1
        if not self.processor.tokenizer.pad_token:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        if not "t5" in model or not "t0" in model:
            self.processor.tokenizer.padding_side = "left"
        if "fuyu" in self.model_name:
            self.processor.image_processor.size = dict(width=720, height=720) # patch size 30 so this is like 224/14
        # if "instructblip-vicuna" in model:
        #     self.model.config.text_config.pad_token_id = self.processor.tokenizer.pad_token_id
        #     self.model.language_model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.image_root = args.image_root
        self.prompt_query = args.prompt_query  # Which of these choices is shown in the image?
        self.task = args.task

    def generate(self, batch):
        if "instructblip-vicuna" in self.model_name:
            generation = self.model.generate(**batch, do_sample=False, max_new_tokens=10, pad_token_id=self.processor.tokenizer.pad_token_id)
        else:
            generation = self.model.generate(**batch, do_sample=False, max_new_tokens=100, min_new_tokens=1)
        captions = self.processor.batch_decode(generation, skip_special_tokens=True)
        if "ASSISTANT:" in captions[0]:
            captions = [c.split("ASSISTANT:")[1].strip() for c in captions]
        elif "[/INST]" in captions[0]:
            captions = [c.split("[/INST]")[1].strip() for c in captions]
        elif "<|assistant|>" in captions[0]:
            captions = [c.split("<|assistant|>")[1].strip() for c in captions]
        elif "Assistant:" in captions[0]:
            captions = [c.split("Assistant:")[1].strip() for c in captions]
        elif "fuyu" in self.model_name:
            captions = [c.split("")[1].strip() for c in captions]
        elif "pali" in self.model_name:
            captions = [c.split("directly.\n")[1].strip() for c in captions]
        elif "Phi-3-vision" in self.model_name:
            captions = [c.split("directly. \n")[1].strip() for c in captions]
        return captions


    def collate(self, batch):
        options = [b["options"] for b in batch]
        prompts_labels_mapping = [prepare_prompt(self.model_name, option, self.task, prompt_query=self.prompt_query) for option in options]
        prompts, labels, mapping = list(zip(*prompts_labels_mapping))
        image_files = [b["image"] for b in batch]
        images = [Image.open(os.path.join(self.image_root, b["image"])).convert('RGB') for b in batch]
        # list(prompts) because normally prompts are tuples but at least FuyuProcessor NEEDS list.
        if "Phi-3-vision" in self.model_name:
            assert len(prompts) == 1
            inputs = self.processor(text=prompts[0], images=images, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(text=list(prompts), images=images, return_tensors="pt", padding=True)
        return inputs, labels, mapping, image_files


class IdeficsModel(HFModel):
    def __init__(self, args):
        model = args.model
        load_quantized = args.load_quantized
        cache_dir = args.model_cache_dir

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_quantized == "4bit",
            load_in_8bit=load_quantized == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = IdeficsForVisionText2Text.from_pretrained(model,
                                                          quantization_config=bnb_config if load_quantized else None,
                                                          torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                                                          trust_remote_code=True)
        if not bnb_config.load_in_4bit and not bnb_config.load_in_8bit:
            self.model = self.model.to("cuda")

        self.model_name = model
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.image_root = args.image_root
        self.prompt_query = args.prompt_query  # Which of these choices is shown in the image?
        self.task = args.task

    def generate(self, batch):
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"],
                                            add_special_tokens=False).input_ids

        generation = self.model.generate(**batch, eos_token_id=exit_condition, bad_words_ids=bad_words_ids,
                                         do_sample=False, max_new_tokens=10)

        captions = self.processor.batch_decode(generation, skip_special_tokens=True)
        captions = [c.split("Assistant:")[1].strip() for c in captions]
        return captions

    def collate(self, batch):
        options = [b["options"] for b in batch]
        prompts_labels_mapping = [prepare_prompt(self.model_name, option, self.task, prompt_query=self.prompt_query) for option in options]
        prompts, labels, mapping = list(zip(*prompts_labels_mapping))
        image_files = [b["image"] for b in batch]
        images = [Image.open(os.path.join(self.image_root, b["image"])) for b in batch]
        idefics_prompts = []
        for prompt, image in zip(prompts, images):
            idefics_prompts.append([
                image,
                prompt,
                "<end_of_utterance>",
                "\nAssistant:"
            ])

        inputs = self.processor(idefics_prompts, add_end_of_utterance_token=False, return_tensors="pt", padding=True)
        return inputs, labels, mapping, image_files



class Idefics2Model(HFModel):
    def __init__(self, args):
        model = args.model
        load_quantized = args.load_quantized
        cache_dir = args.model_cache_dir

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_quantized == "4bit",
            load_in_8bit=load_quantized == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForVision2Seq.from_pretrained(model,
                                                          quantization_config=bnb_config if load_quantized else None,
                                                            # ssss_attn_implementation="flash_attention_2",
                                                          torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                                                            low_cpu_mem_usage=True)
        if not bnb_config.load_in_4bit and not bnb_config.load_in_8bit:
            self.model = self.model.to("cuda")

        self.model_name = model
        self.processor = AutoProcessor.from_pretrained(model, do_image_splitting=False )
        self.image_root = args.image_root
        self.prompt_query = args.prompt_query  # Which of these choices is shown in the image?
        self.task = args.task

    def generate(self, batch):

        generation = self.model.generate(**batch,
                                         do_sample=False, max_new_tokens=1000)

        captions = self.processor.batch_decode(generation, skip_special_tokens=True)
        captions = [c.split("Assistant:")[1].strip() for c in captions]
        captions = [c.replace("Answer: ", "") for c in captions]
        
        return captions

    def collate(self, batch):
        options = [b["options"] for b in batch]
        prompts_labels_mapping = [prepare_prompt(self.model_name, option, self.task, prompt_query=self.prompt_query) for option in options]
        prompts, labels, mapping = list(zip(*prompts_labels_mapping))
        image_files = [b["image"] for b in batch]
        images = [[Image.open(os.path.join(self.image_root, b["image"]))] for b in batch]
        idefics_prompts = []

        messages = [[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]
            }] for prompt in prompts]

        prompts = [self.processor.apply_chat_template(message, add_generation_prompt=True) for message in messages]
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)

        return inputs, labels, mapping, image_files


class QwenVLModel(HFModel):
    def __init__(self, args):
        model = args.model
        load_quantized = args.load_quantized
        cache_dir = args.model_cache_dir

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_quantized == "4bit",
            load_in_8bit=load_quantized == "8bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(model,
                                                          # quantization_config=bnb_config if load_quantized else None,
                                                          torch_dtype=torch.bfloat16, cache_dir=cache_dir,
                                                          trust_remote_code=True)
        if not bnb_config.load_in_4bit and not bnb_config.load_in_8bit:
            self.model = self.model.to("cuda")

        self.model_name = model
        self.processor = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.image_root = args.image_root
        self.prompt_query = args.prompt_query  # Which of these choices is shown in the image?
        self.task = args.task
        self.processor.pad_token = "<|endoftext|>"
        self.processor.pad_token_id = self.processor.special_tokens["<|endoftext|>"]
        self.processor.padding_side = "left"

    def generate(self, batch):
        # batch = batch.batch
        # response, history = self.model.chat(self.processor, query=batch)
        generation = self.model.generate(**batch, do_sample=False, max_new_tokens=10)
        captions = self.processor.batch_decode(generation, skip_special_tokens=True)
        if "assistant" in captions[0]:
            captions = [c.split("assistant")[1].strip() for c in captions]
        return captions

    def collate(self, batch):
        options = [b["options"] for b in batch]
        prompts_labels_mapping = [prepare_prompt(self.model_name, option, self.task, prompt_query=self.prompt_query) for option in options]
        prompts, labels, mapping = list(zip(*prompts_labels_mapping))
        image_files = [b["image"] for b in batch]
        images = [os.path.join(self.image_root, b["image"]) for b in batch]
        query = [self.processor.from_list_format([{'image': img},{'text': prompt}]) for img, prompt in zip(images, prompts)]
        inputs = self.processor(query, return_tensors="pt", padding=True)
        return inputs, labels, mapping, image_files


class InternLMBatch:
    def __init__(self, images, prompts):
        self.images = images
        self.prompts = prompts

    def to(self, *args, **kwargs):
        return self

class InternLMXComposerModel(HFModel):
    def __init__(self, args):
        model = args.model
        load_quantized = args.load_quantized
        cache_dir = args.model_cache_dir

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_quantized == "4bit",
            load_in_8bit=load_quantized == "8bit",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = InternLMXComposer2ForCausalLM.from_pretrained(model,
                                                          quantization_config=bnb_config if load_quantized else None,
                                                          torch_dtype=torch.float16,
                                                          cache_dir=cache_dir,
                                                          trust_remote_code=True)
        if not bnb_config.load_in_4bit and not bnb_config.load_in_8bit:
            self.model = self.model.to("cuda")

        self.model_name = model
        self.processor = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.vis_processor = transforms.Compose([
            transforms.Resize((self.model.config.img_size, self.model.config.img_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.image_root = args.image_root
        self.prompt_query = args.prompt_query  # Which of these choices is shown in the image?
        self.task = args.task
        # self.processor.pad_token = "<|endoftext|>"
        # self.processor.pad_token_id = self.processor.special_tokens["<|endoftext|>"]
        self.processor.padding_side = "left"

    def generate(self, batch):
        images = torch.stack(batch.images)
        questions = batch.prompts
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                images = self.model.encode_img(images)

                inputs_list = []
                masks_list = []
                max_len = 0
                for image, question in zip(images, questions):
                    inputs, im_mask = self.model.interleav_wrap_chat(self.processor, "<ImageHere>" + question, image.unsqueeze(0),
                                                                [], "")
                    inputs_list.append(inputs)
                    masks_list.append(im_mask)
                    max_len = max(max_len, im_mask.shape[1])

                pad_embed = self.model.model.tok_embeddings(torch.tensor(self.processor.pad_token_id).cuda()).unsqueeze(
                    0).unsqueeze(0)
                batch_inputs, batch_masks, batch_atten_masks = [], [], []
                for inputs, im_mask in zip(inputs_list, masks_list):
                    if im_mask.shape[1] < max_len:
                        pad_embeds = torch.cat([pad_embed] * (max_len - im_mask.shape[1]), dim=1)
                        pad_masks = torch.tensor([0] * (max_len - im_mask.shape[1])).unsqueeze(0).cuda()
                        inputs = torch.cat([pad_embeds, inputs['inputs_embeds']], dim=1)
                        atten_masks = torch.cat([pad_masks, torch.ones_like(im_mask)], dim=1)
                        im_mask = torch.cat([pad_masks, im_mask], dim=1)
                    else:
                        inputs = inputs['inputs_embeds']
                        atten_masks = torch.ones_like(im_mask)

                    batch_inputs.append(inputs)
                    batch_masks.append(im_mask)
                    batch_atten_masks.append(atten_masks)

                batch_inputs = {'inputs_embeds': torch.cat(batch_inputs, dim=0)}
                batch_masks = torch.cat(batch_masks, dim=0).bool()
                batch_atten_masks = torch.cat(batch_atten_masks, dim=0).bool()

                print(batch_inputs['inputs_embeds'].shape, batch_masks.shape)
                eos_token_id = [
                    self.processor.eos_token_id,
                    self.processor.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0]
                ]

                generation = self.model.generate(
                            **batch_inputs,
                            im_mask=batch_masks,
                            attention_mask=batch_atten_masks,
                            do_sample=False, max_new_tokens=10,
                            eos_token_id=eos_token_id)
        captions = self.processor.batch_decode(generation, skip_special_tokens=True)
        captions = [c.split('[UNUSED_TOKEN_145]')[0].strip() for c in captions]
        captions = [c.split("The answer is ")[1] if "The answer is" in c else c for c in captions]
        return captions

    def collate(self, batch):
        options = [b["options"] for b in batch]
        prompts_labels_mapping = [prepare_prompt(self.model_name, option, self.task, prompt_query=self.prompt_query) for option in options]
        prompts, labels, mapping = list(zip(*prompts_labels_mapping))
        image_files = [b["image"] for b in batch]
        images = [self.vis_processor(Image.open(os.path.join(self.image_root, b["image"])).convert("RGB")) for b in batch]
        batch = InternLMBatch(images=images, prompts=prompts)
        return batch, labels, mapping, image_files