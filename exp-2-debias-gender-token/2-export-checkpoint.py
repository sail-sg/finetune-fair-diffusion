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

import os
from pathlib import Path

import argparse
import itertools
import logging
import pytz
import random
import copy
import yaml
from typing import Optional, Tuple, Union

import torch
from torch import nn

import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, GradScalerKwargs

import diffusers
from diffusers import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel



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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script to finetune Stable Diffusion for debiasing purposes.")

    # 1. experiment setting
    parser.add_argument(
        '--proj_name', 
        default="debias-SD",
        help="proj name",
        type=str, 
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_num_tokens",
        type=int,
        default=5,
        help="number of tokens to finetune as prompt prefix",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default="5991", 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20,
        help=(
            "Save a temporary checkpoint every X steps. "
            "The purpose of these checkpoints is to easily resume training "
            "when some error occurs during training."
        )
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=(
            "Max number of temporary checkpoints to store. "
            "The oldest ones will be deleted when new checkpoints are saved."),
    )
    parser.add_argument(
        "--checkpointing_steps_long",
        type=int,
        default=200,
        help=(
            "Save a checkpoint every Y steps. "
            "These checkpoints will not be deleted. They are used for final evaluation. "
        ),
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
    # parser.add_argument(
    #     "--enable_xformers_memory_efficient_attention", 
    #     action="store_true", 
    #     default=True,
    #     help="Whether or not to use xformers."
    # )
    parser.add_argument(
        "--rank",
        type=int,
        default=50,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        '--train_plot_every_n_iter', 
        help="plot training stats every n iteration", 
        type=int, 
        default=20
        )
    parser.add_argument(
        '--evaluate_every_n_iter', 
        help="evaluate every n iteration", 
        type=int,
        default=200
        )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help='only `"wandb"` is supported',
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        '--guidance_scale', 
        help="diffusion model text guidance scale", 
        type=float, 
        default=7.5
        )
    parser.add_argument(
        '--EMA_decay', 
        help="decay coefficient for EMA",
        type=float,
        default=0.996
        )

    # loss weight
    parser.add_argument(
        '--weight_loss_img', 
        default=8,
        help="weight for the image semantics preserving loss", 
        type=float, 
    )
    parser.add_argument(
        '--weight_loss_face', 
        default=1,
        help="weight for the face realism preserving loss", 
        type=float, 
    )
    parser.add_argument(
        '--uncertainty_threshold', 
        help="the uncertainty threshold used in distributional alignment loss", 
        type=float, 
        default=0.2
        )
    parser.add_argument('--factor1', help="train, val, test batch size", type=float, default=0.2)
    parser.add_argument('--factor2', help="train, val, test batch size", type=float, default=0.2)

    # batch size, properly set to max out GPU
    parser.add_argument(
        '--train_images_per_prompt_GPU', 
        help=(
            "number of images generated for a prompt per GPU during training. "
            "These images are used as a batch for distributional alignment."
        ), 
        type=int, 
        default=8,
        )
    parser.add_argument(
        '--train_GPU_batch_size', 
        help="training batch size in every GPU", 
        type=int, 
        default=4
        )
    parser.add_argument(
        '--val_images_per_prompt_GPU', 
        help=(
            "number of images generated for a prompt per GPU during validation. "
            "These images are used to measure bias."
        ),
        type=int, 
        default=8
        )
    parser.add_argument(
        '--val_GPU_batch_size', 
        help="validation batch size in every GPU", 
        type=int, 
        default=8
        )    


    # experiment input and output paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="logs will be saved to args.output_dir/args.proj_name/args.logging_dir",
    )
    parser.add_argument(
        "--prompt_occupation_path",
        type=str,
        default="../data/1-prompts/occupation.json",
        help="prompt template, and occupations for train and val",
    )
    parser.add_argument(
        '--classifier_weight_path', 
        default="../data/2-trained-classifiers/CelebA_MobileNetLarge_08060852/epoch=9-step=12660_MobileNetLarge.pt",
        help="pre-trained classifer that predicts binary gender", 
        type=str,
        required=False, 
    )
    parser.add_argument(
        '--face_feats_path', 
        help="external face feats, used for the face realism preserving loss", 
        type=str, 
        default="../data/3-face-features/CelebA_MobileNetLarge_08240859/face_feats.pkl"
        )
    # parser.add_argument(
    #     '--aligned_face_gender_model_path', 
    #     help="train, val, test batch size", 
    #     type=str, 
    #     default="../data/3-face-features/CelebA_MobileNetLarge_08240859/epoch=9-step=6330_MobileNetLarge.pt"
    #     )
    parser.add_argument('--opensphere_config', help="train, val, test batch size", type=str, default="../data/4-opensphere_checkpoints/opensphere_checkpoints/20220424_210641/config.yml")
    parser.add_argument('--opensphere_model_path', help="train, val, test batch size", type=str, default="../data/4-opensphere_checkpoints/opensphere_checkpoints/20220424_210641/models/backbone_100000.pth")

    # learning related settings
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", 
        type=float, 
        default=1.0, 
        help="Power factor of the polynomial scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=100.0, type=float, help="Max gradient norm.")

    # settings that should not be changed
    # we didn't experiment with other values
    parser.add_argument(
        "--img_size_small",
        type=int,
        default=224,
        help="For some operations, images will be resized to this size for more efficient processing",
    )
    parser.add_argument(
        "--size_face",
        type=int,
        default=224,
        help="faces will be resized to this size",
    )
    parser.add_argument(
        "--size_aligned_face",
        type=int,
        default=112,
        help="aligned faces will be resized to this size",
    )
    parser.add_argument('--face_gender_confidence_level', help="train, val, test batch size", type=float, default=0.9)

    # passed directly by accelerate
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
        )
    
    # config file
    parser.add_argument("--config", help="config file", type=str, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)
        args_dict = vars(args)
        for key, value in config_data.items():
            args_dict[key] = type(args_dict[key])(value)
        args = argparse.Namespace(**args_dict)

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

logger = get_logger(__name__)

def main(args):

    logging_dir = Path(args.output_dir, args.logging_dir)

    kwargs = GradScalerKwargs(
        init_scale = 2.**0,
        growth_interval=99999999, 
        backoff_factor=0.5,
        growth_factor=2,
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=1, # we did not implement gradient accumulation
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    set_seed(args.seed, device_specific=True)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
        )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="text_encoder"
        )
    # vae = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="vae",
    #     )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="unet",
        )
    

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # vae.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    # vae.enable_gradient_checkpointing()

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype_high_precision)
    unet.to(accelerator.device, dtype=weight_dtype)
    # vae.to(accelerator.device, dtype=weight_dtype)
    
    prefix_tokens = [f"<common-token{i+1}>" for i in range(args.train_num_tokens)]
    prefix_tokens_id = expand_tokenizer(tokenizer, text_encoder, prefix_tokens)

    prefix_embedding = FairEmbeddings(
        text_encoder.text_model.embeddings.token_embedding,
        text_encoder.text_model.embeddings.position_embedding,
        prefix_tokens_id,
        text_encoder.text_model.config.max_position_embeddings,
        dtype=weight_dtype_high_precision
    )
    prefix_embedding.to(accelerator.device)
    prefix_embedding_ema = EMAModel(prefix_embedding.token_embedding.parameters(), decay=args.EMA_decay)
    prefix_embedding_ema.to(accelerator.device)
    
    prompt_debiaser = lambda x: "".join(prefix_tokens) + x
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    params_to_optimize = prefix_embedding.token_embedding.parameters()
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    optimizer, lr_scheduler = accelerator.prepare(
            optimizer, lr_scheduler
        )
    
    _ = accelerator.prepare(prefix_embedding)
    prefix_embedding_ema = accelerator.prepare(prefix_embedding_ema)
    accelerator.register_for_checkpointing(prefix_embedding_ema)

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        raise ValueError("resume_from_checkpoint must be provided.")
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            raise ValueError(f"{args.resume_from_checkpoint}' does not exist.")

        args.export_dir = str(Path(args.resume_from_checkpoint).parent / (Path(args.resume_from_checkpoint).name + "_exported"))
        if not os.path.exists(args.export_dir):
            os.makedirs(args.export_dir)

        accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)

        prefix_embedding_ema.to(accelerator.device)


        prefix_embedding_save_path = os.path.join(args.export_dir, "prefix_embedding.pth")
        torch.save(prefix_embedding.cpu().state_dict(), prefix_embedding_save_path)
        accelerator.print(f"prefix_embedding weight saved to {prefix_embedding_save_path}.")

        prefix_embedding_ema_dict = copy.deepcopy( prefix_embedding.state_dict() )
        prefix_embedding_ema_dict["token_embedding.weight"] = prefix_embedding_ema.shadow_params[0]
        prefix_embedding_ema_dict_cpu = {key: value.to('cpu') for key, value in prefix_embedding_ema_dict.items()}

        prefix_embedding_ema_save_path = os.path.join(args.export_dir, "prefix_embedding_EMA.pth")
        torch.save(prefix_embedding_ema_dict_cpu, prefix_embedding_ema_save_path)
        accelerator.print(f"prefix_embedding EMA weight saved to {prefix_embedding_ema_save_path}.")

        import pdb; pdb.set_trace()




if __name__ == "__main__":
    args = parse_args()
    main(args)


# python 2-export-checkpoint.py --config configs/debias-token.yaml --resume_from_checkpoint ./outputs/from-paper_finetune-prefix_09202102/checkpoint-9400