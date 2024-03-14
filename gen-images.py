#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import itertools
import math
import os


import torch
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
import json
import pytz

from torch import nn
from torchvision import transforms
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling



# os.environ["gpu_ids"] = "1"
my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.


class FairEmbeddings(nn.Module):
    def __init__(self, token_embedding, position_embedding, fair_tokens, max_position_embeddings, dtype=torch.float32):
        super().__init__()
        
        self.position_embedding = position_embedding.to(dtype)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        
        self.token_embedding = nn.Embedding(len(fair_tokens)+1, token_embedding.embedding_dim).to(dtype)
        self.token_embedding.weight.data.zero_()
        self.token_embedding.weight.data[1:,:] = token_embedding.weight.data[fair_tokens]
        
        self.fair_token_id_map = {}
        for key in range( token_embedding.weight.shape[0] ):
            self.fair_token_id_map[key] = 0
        for val, key in enumerate(fair_tokens):
            self.fair_token_id_map[key] = val + 1

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        unfair_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        input_ids_fair = input_ids.cpu().apply_(self.fair_token_id_map.get).to(input_ids.device)
        inputs_embeds = self.token_embedding(input_ids_fair)

        position_embeddings = self.position_embedding(position_ids)
        fair_embeddings = inputs_embeds + position_embeddings

        unfair_embeds[input_ids_fair!=0] = fair_embeddings[input_ids_fair!=0]
        return unfair_embeds

def expand_tokenizer(tokenizer, text_encoder, placeholder_token):
    # Add the placeholder_token in tokenizer
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
    # Convert the initializer_token, placeholder_token to ids

    random_token_idx = list(range(text_encoder.get_input_embeddings().weight.shape[0]))
    random.shuffle(random_token_idx)
    random_token_idx = random_token_idx[:len(placeholder_token_id)]
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    for idx, random_idx in itertools.zip_longest(placeholder_token_id,random_token_idx):
        token_embeds[idx] = token_embeds[random_idx]

    return placeholder_token_id


@torch.no_grad()
def generate_image(prompt, noises, tokenizer, text_encoder, unet, vae, noise_scheduler, num_denoising_steps=30, guidance_scale=7.5, device="cuda:0", weight_dtype=torch.float16, weight_dtype_high_precision=torch.float32):
    """
    prompts: str
    noises: [N,4,64,64], N is number images to be generated for the prompt
    """
    N = noises.shape[0]
    prompts = [prompt] * N
    
    prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
    prompts_token["input_ids"] = prompts_token["input_ids"].to(device)
    prompts_token["attention_mask"] = prompts_token["attention_mask"].to(device)

    prompt_embeds = text_encoder(
        prompts_token["input_ids"],
        prompts_token["attention_mask"],
    )
    prompt_embeds = prompt_embeds[0]

    batch_size = prompt_embeds.shape[0]
    uncond_tokens = [""] * batch_size
    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
    uncond_input["input_ids"] = uncond_input["input_ids"].to(device)
    uncond_input["attention_mask"] = uncond_input["attention_mask"].to(device)
    negative_prompt_embeds = text_encoder(
        uncond_input["input_ids"],
        uncond_input["attention_mask"],
    )
    negative_prompt_embeds = negative_prompt_embeds[0]

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds = prompt_embeds.to(weight_dtype)
    
    noise_scheduler.set_timesteps(num_denoising_steps)
    latents = noises
    for i, t in enumerate(noise_scheduler.timesteps):
    
        # scale model input
        latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
        noises_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample
        noises_pred = noises_pred.to(weight_dtype_high_precision)
        
        noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
        noises_pred = noises_pred_uncond + guidance_scale * (noises_pred_text - noises_pred_uncond)
        
        latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

    latents = 1 / vae.config.scaling_factor * latents
    images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1,1) # in range [-1,1]
    
    return images


# Copied from transformers.models.clip.modeling_clip.py
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def text_model_forward(
    text_encoder,
    input_ids: Optional[torch.Tensor] = None,
    token_embeds: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    
    # copied from transformers 4.30.0.dev0, transformers/models/clip/modeling_clip.py/CLIPTextTransformer
    def _build_causal_attention_mask(bsz, seq_len, dtype, device=None):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype, device=device)
        mask.fill_(torch.finfo(dtype).min)
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask
    
    output_attentions = output_attentions if output_attentions is not None else text_encoder.text_model.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else text_encoder.text_model.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else text_encoder.text_model.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify input_ids")
    
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    
    # only difference!
    # hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
    hidden_states = token_embeds

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    # TODO: the pooled_output is wrong
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

# def generate_image_w_prefix_embedding(prompt, noises, num_denoising_steps, which_text_encoder, which_unet, which_prefix_embedding):
def generate_image_w_prefix_embedding(prompt, noises, prefix_embedding, tokenizer, text_encoder, unet, vae, noise_scheduler, num_denoising_steps=30, guidance_scale=7.5, device="cuda:0", weight_dtype=torch.float16, weight_dtype_high_precision=torch.float32):
    """
    prompts: str
    noises: [N,4,64,64], N is number images to be generated for the prompt
    """
    pipe = StableDiffusionPipeline(
            vae = vae,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            unet = unet,
            scheduler = noise_scheduler,
            safety_checker = None,
            requires_safety_checker=False,
            feature_extractor = None,
        )
    
    N = noises.shape[0]
    prompts = [prompt] * N
    
    prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
    prompts_token["input_ids"] = prompts_token["input_ids"].to(device)
    prompts_token["attention_mask"] = prompts_token["attention_mask"].to(device)

    # encoder_hidden_states = self.text_encoder(prompts_token["input_ids"], prompts_token["attention_mask"])[0]
    # the following code segment does exactly the same thing as the above line
    token_embeds = text_encoder.text_model.embeddings(
        input_ids=prompts_token["input_ids"],
        position_ids=None
        )
    token_embeds = prefix_embedding(
        input_ids=prompts_token["input_ids"],
        position_ids=None,
        unfair_embeds=token_embeds,
    )
    # token_embeds = token_embeds.to(weight_dtype)
    encoder_hidden_states = text_model_forward(
        text_encoder,
        prompts_token["input_ids"],
        token_embeds,
        prompts_token["attention_mask"]
        )[0]
    prompt_embeds = pipe._encode_prompt(
        prompt=None,
        device=noises.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        prompt_embeds=encoder_hidden_states,
        )
        
    prompt_embeds = prompt_embeds.to(weight_dtype)
        
    noise_scheduler.set_timesteps(num_denoising_steps)
    latents = noises
    for i, t in enumerate(noise_scheduler.timesteps):
    
        # scale model input
        latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
        noises_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample
        noises_pred = noises_pred.to(weight_dtype_high_precision)
        
        noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
        noises_pred = noises_pred_uncond + guidance_scale * (noises_pred_text - noises_pred_uncond)
        
        latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

    latents = 1 / vae.config.scaling_factor * latents
    images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1,1) # in range [-1,1]
    
    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script to finetune Stable Diffusion for debiasing purposes.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_text_encoder_lora_from",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--load_unet_lora_from",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--load_prefix_embedding_from", 
        type=str,
        default=None,
    )
    parser.add_argument(
        "--number_prefix_tokens",
        type=int,
        default=5,
        help="number of tokens as prefix, must be provided when --load_prefix_embedding_from is provided",
    )
    parser.add_argument(
        "--gpu_id", 
        type=int,
        default=0,
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_imgs_per_prompt",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1997,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="provide the checkpoint path to resume from checkpoint",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=50,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        '--guidance_scale', 
        help="diffusion model text guidance scale", 
        type=float, 
        default=7.5
        )
    parser.add_argument(
        '--num_denoising_steps', 
        help="num denoising steps used for image generation", 
        type=int, 
        default=30
        )
    parser.add_argument(
        '--batch_size', 
        help="batch size for image generation", 
        type=int, 
        default=10
        )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):

    args.device = f"cuda:{args.gpu_id}"

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
        )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder"
        )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet",
        )
    noise_scheduler = DPMSolverMultistepScheduler.from_config(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler",
        )

    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    text_encoder.to(args.device, dtype=weight_dtype)
    unet.to(args.device, dtype=weight_dtype)
    vae.to(args.device, dtype=weight_dtype)

    if args.load_text_encoder_lora_from:
        text_encoder_lora_params = LoraLoaderMixin._modify_text_encoder(text_encoder, dtype=torch.float32, rank=args.rank, patch_mlp=True)

        text_encoder_lora_dict = torch.load(args.load_text_encoder_lora_from, map_location=args.device)
        _ = text_encoder.load_state_dict(text_encoder_lora_dict, strict=False)

    if args.load_unet_lora_from:
        unet_lora_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            unet_lora_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=args.rank,
            ).to(args.device)
            
        unet.set_attn_processor(unet_lora_procs)

        unet_lora_dict = torch.load(args.load_unet_lora_from, map_location=args.device)
        _ = unet.load_state_dict(unet_lora_dict, strict=False)
    
    if args.load_prefix_embedding_from:
        prefix_tokens = [f"<common-token{i+1}>" for i in range(args.number_prefix_tokens)]
        prefix_tokens_id = expand_tokenizer(tokenizer, text_encoder, prefix_tokens)
        prompt_debiaser = lambda x: "".join(prefix_tokens) + x

        prefix_embedding = FairEmbeddings(
            text_encoder.text_model.embeddings.token_embedding,
            text_encoder.text_model.embeddings.position_embedding,
            prefix_tokens_id,
            text_encoder.text_model.config.max_position_embeddings,
            dtype=weight_dtype_high_precision
        )
        prefix_embedding.to(args.device)

        prefix_embedding_dict = torch.load(args.load_prefix_embedding_from, map_location=args.device)
        _ = prefix_embedding.load_state_dict(prefix_embedding_dict, strict=False)


    # Dataset and DataLoaders creation:
    with open(args.prompts_path, 'r') as f:
        experiment_data = json.load(f)
    test_prompts = experiment_data["test_prompts"]
    
    # generate noise in this way so that for every prompt, the first, second, third, ..., generated images are always generated using the same noise
    # even when args.num_imgs_per_prompt changes
    noise_all = []
    for prompt in test_prompts:
        noise_per_prompt = []
        for i in range(args.num_imgs_per_prompt):
            torch.manual_seed(args.random_seed + hash(prompt) + i)
            noise_single = torch.randn([1,4,64,64], dtype=weight_dtype_high_precision).to(args.device)
            noise_per_prompt.append(noise_single)
        noise_per_prompt = torch.cat(noise_per_prompt).unsqueeze(0)
        noise_all.append(noise_per_prompt)
    noise_all = torch.cat(noise_all)
    
    for i, prompt_i in tqdm(enumerate(test_prompts), total=len(test_prompts), desc='Prompts', leave=True):
        
        save_dir_prompt_i = os.path.join(args.save_dir, f"prompt_{i}")
        os.makedirs(save_dir_prompt_i, exist_ok=True)

        noises_to_use = []
        img_save_paths_to_use = []
        for j in range(args.num_imgs_per_prompt):
            img_save_path = os.path.join( save_dir_prompt_i, f"img_{j}.jpg" )
            if not os.path.exists(img_save_path):
                noises_to_use.append(noise_all[i,j].unsqueeze(dim=0))
                img_save_paths_to_use.append(img_save_path)
        noises_to_use = torch.cat(noises_to_use)

        N = math.ceil(noises_to_use.shape[0] / args.batch_size)
        for j in tqdm(range(N), desc='Images per prompt', leave=False):
            noises_ij = noises_to_use[args.batch_size*j:args.batch_size*(j+1)]
            img_save_paths_ij = img_save_paths_to_use[args.batch_size*j:args.batch_size*(j+1)]

            if args.load_prefix_embedding_from:
                images_ij = generate_image_w_prefix_embedding(
                    prompt_debiaser(prompt_i), 
                    noises_ij,
                    prefix_embedding=prefix_embedding,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder, 
                    unet=unet,
                    vae=vae, 
                    noise_scheduler=noise_scheduler, 
                    num_denoising_steps=args.num_denoising_steps, 
                    guidance_scale=args.guidance_scale, 
                    device=args.device, 
                    weight_dtype=weight_dtype, 
                    weight_dtype_high_precision=weight_dtype_high_precision
                    )
            else:
                images_ij = generate_image(
                    prompt_i, 
                    noises_ij,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder, 
                    unet=unet,
                    vae=vae, 
                    noise_scheduler=noise_scheduler, 
                    num_denoising_steps=args.num_denoising_steps, 
                    guidance_scale=args.guidance_scale, 
                    device=args.device, 
                    weight_dtype=weight_dtype, 
                    weight_dtype_high_precision=weight_dtype_high_precision
                    )
            
            for img, img_save_path in itertools.zip_longest(images_ij, img_save_paths_ij):
                img_pil = transforms.ToPILImage()(img*0.5+0.5)
                img_pil.save(img_save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)




"""
1. python gen-images.py \
    --load_text_encoder_lora_from ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200_exported/text_encoder_lora_EMA.pth \
    --prompts_path ./data/1-prompts/occupation.json \
    --num_imgs_per_prompt 49 \
    --save_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_occupation/ \
    --gpu_id 0 \
    --batch_size 10

2. python gen-images.py \
    --load_text_encoder_lora_from ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200_exported/text_encoder_lora_EMA.pth \
    --prompts_path ./data/1-prompts/occupation_w_style_and_context.json \
    --num_imgs_per_prompt 49 \
    --save_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_occupation_w_style_and_context/ \
    --gpu_id 0 \
    --batch_size 10

3. python gen-images.py \
    --load_text_encoder_lora_from ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200_exported/text_encoder_lora_EMA.pth \
    --prompts_path ./data/1-prompts/personal_descriptor.json \
    --num_imgs_per_prompt 49 \
    --save_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_personal_descriptor/ \
    --gpu_id 0 \
    --batch_size 10

4. python gen-images.py \
    --load_text_encoder_lora_from ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200_exported/text_encoder_lora_EMA.pth \
    --prompts_path ./data/1-prompts/sports.json \
    --num_imgs_per_prompt 49 \
    --save_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_sports/ \
    --gpu_id 0 \
    --batch_size 10
"""






"""
1. python gen-images.py \
    --prompts_path ./data/1-prompts/occupation.json \
    --num_imgs_per_prompt 49 \
    --save_dir ./original-SD-generated-images/test_prompts_occupation/ \
    --gpu_id 0 \
    --batch_size 10

2. python gen-images.py \
    --prompts_path ./data/1-prompts/occupation_w_style_and_context.json \
    --num_imgs_per_prompt 49 \
    --save_dir ./original-SD-generated-images/test_prompts_occupation_w_style_and_context/ \
    --gpu_id 1 \
    --batch_size 10

3. python gen-images.py \
    --prompts_path ./data/1-prompts/personal_descriptor.json \
    --num_imgs_per_prompt 49 \
    --save_dir ./original-SD-generated-images/test_prompts_personal_descriptor/ \
    --gpu_id 0 \
    --batch_size 10

4. python gen-images.py \
    --prompts_path ./data/1-prompts/sports.json \
    --num_imgs_per_prompt 49 \
    --save_dir ./original-SD-generated-images/test_prompts_sports/ \
    --gpu_id 1 \
    --batch_size 10
"""