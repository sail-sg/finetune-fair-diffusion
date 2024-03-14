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

import os, sys
from pathlib import Path

import argparse
import itertools
import logging
import math
import shutil
import json
import pytz
import random
from datetime import datetime
from tqdm.auto import tqdm
import copy
import pickle as pkl
import yaml
from packaging import version
from PIL import Image, ImageOps, ImageDraw, ImageFont
from typing import Optional, Tuple, Union

import numpy as np
import scipy
from skimage import transform
import kornia
from sentence_transformers import util

import torch
from torch import nn
import torchvision
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch.utils.data import Dataset
from torchvision import transforms

import transformers
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.modeling_outputs import BaseModelOutputWithPooling
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, GradScalerKwargs

import diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from diffusers.loaders import (
    LoraLoaderMixin,
)
from diffusers.models.attention_processor import (
    LoRAAttnProcessor,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.training_utils import EMAModel

# you MUST import torch before insightface
# otherwise onnxruntime, used by FaceAnalysis, can only use CPU
from insightface.app import FaceAnalysis



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

class FaceFeatsModel(torch.nn.Module):
    def __init__(self, face_feats_path):
        super().__init__()
        
        with open(face_feats_path, "rb") as f:
            face_feats, face_genders, face_logits = pkl.load(f)
        
        face_feats = torch.nn.functional.normalize(face_feats, dim=-1)
        self.face_feats = nn.Parameter(face_feats)   
        self.face_feats.requires_grad_(False)        
        
    def forward(self, x):
        """no forward function
        """
        return None
        
    @torch.no_grad()
    def semantic_search(self, query_embeddings, selector=None, return_similarity=False):
        """search the closest face embedding from vector database.
        """
        target_embeddings = torch.ones_like(query_embeddings) * (-1)
        if return_similarity:
            similarities = torch.ones([query_embeddings.shape[0]], device=query_embeddings.device, dtype=query_embeddings.dtype) * (-1)
            
        if selector.sum()>0:
            hits = util.semantic_search(query_embeddings[selector], self.face_feats, score_function=util.dot_score, top_k=1)
            target_embeddings_ = torch.cat([self.face_feats[hit[0]["corpus_id"]].unsqueeze(dim=0) for hit in hits])
            target_embeddings[selector] = target_embeddings_
            if return_similarity:
                similarities_ = torch.tensor([hit[0]["score"] for hit in hits], device=query_embeddings.device, dtype=query_embeddings.dtype)
                similarities[selector] = similarities_
        
        if return_similarity:
            return target_embeddings.data.detach().clone(), similarities
        else:
            return target_embeddings.data.detach().clone()


def clean_checkpoint(ckpts_save_dir, name, checkpoints_total_limit):
    checkpoints = os.listdir(ckpts_save_dir)
    checkpoints = [d for d in checkpoints if d.startswith(name)]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
    if len(checkpoints) >= checkpoints_total_limit:
        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
        removing_checkpoints = checkpoints[0:num_to_remove]

        logger.info(
            f"chekpoint name:{name}, {len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
        )
        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(args.ckpts_save_dir, removing_checkpoint)
            shutil.rmtree(removing_checkpoint)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def plot_in_grid(images, save_to, face_indicators=None, face_bboxs=None, preds_gender=None, pred_class_probs_gender=None):
    """
    images: torch tensor in shape of [N,3,H,W], in range [-1,1]
    """
    images_w_face = images[face_indicators]
    images_wo_face = images[face_indicators.logical_not()]

    # first reorder everything from most to least male, from most to least female, and finally images without faces
    idxs_male = (preds_gender == 1).nonzero(as_tuple=False).view([-1])
    probs_male = pred_class_probs_gender[idxs_male]
    idxs_male = idxs_male[probs_male.argsort(descending=True)]

    idxs_female = (preds_gender == 0).nonzero(as_tuple=False).view([-1])
    probs_female = pred_class_probs_gender[idxs_female]
    idxs_female = idxs_female[probs_female.argsort(descending=True)]

    idxs_no_face = (preds_gender == -1).nonzero(as_tuple=False).view([-1])

    images_to_plot = []
    idxs_reordered = torch.torch.cat([idxs_male, idxs_female, idxs_no_face])
    
    for idx in idxs_reordered:
        img = images[idx]
        face_indicator = face_indicators[idx]
        face_bbox = face_bboxs[idx]
        pred_gender = preds_gender[idx]
        pred_class_prob_gender = pred_class_probs_gender[idx]
        
        if pred_gender == 1:
            pred = "Male"
            border_color = "blue"
        elif pred_gender == 0:
            pred = "Female"
            border_color = "red"
        elif pred_gender == -1:
            pred = "Undetected"
            border_color = "white"
        
        img_pil = transforms.ToPILImage()(img*0.5+0.5)
        img_pil_draw = ImageDraw.Draw(img_pil)  
        img_pil_draw.rectangle(face_bbox.tolist(), fill =None, outline =border_color, width=4)

        img_pil = ImageOps.expand(img_pil, border=(50,0,0,0),fill=border_color)

        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_gender.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_gender.item())*512)], fill ="white", outline =None)

        fnt = ImageFont.truetype(font="../data/0-utils/arial-bold.ttf", size=100)
        img_pil_draw.text((400, 400), f"{idx.item()}", align ="left", font=fnt)

        img_pil = ImageOps.expand(img_pil_draw._image, border=(10,10,10,10),fill="black")
        
        images_to_plot.append(img_pil)
        
    N_imgs = len(images_to_plot)
    N1 = int(math.sqrt(N_imgs))
    N2 = math.ceil(N_imgs / N1)

    for i in range(N1*N2-N_imgs):
        images_to_plot.append(
            Image.new('RGB', color="white", size=images_to_plot[0].size)
        )
    grid = image_grid(images_to_plot, N1, N2)
    if not os.path.exists(os.path.dirname(save_to)):
        os.makedirs(os.path.dirname(save_to))
    grid.save(save_to, quality=25)

def make_grad_hook(coef):
    return lambda x: coef * x

def customized_all_gather(tensor, accelerator, return_tensor_other_processes=False):
    tensor_all = [tensor.detach().clone() for i in range(accelerator.num_processes)]
    torch.distributed.all_gather(tensor_all, tensor)
    if return_tensor_other_processes:
        if accelerator.num_processes>1:
            tensor_others = torch.cat([tensor_all[idx] for idx in range(accelerator.num_processes) if idx != accelerator.local_process_index], dim=0)
        else:
            tensor_others = torch.empty([0,]+ list(tensor_all[0].shape[1:]), device=accelerator.device, dtype=tensor_all[0].dtype)
    tensor_all = torch.cat(tensor_all, dim=0)
    
    if return_tensor_other_processes:
        return tensor_all, tensor_others
    else:
        return tensor_all


def expand_bbox(bbox, expand_coef, target_ratio):
    """
    bbox: [width_small, height_small, width_large, height_large], 
        this is the format returned from insightface.app.FaceAnalysis
    expand_coef: 0 is no expansion
    target_ratio: target img height/width ratio
    
    note that it is possible that bbox is outside the original image size
    confirmed for insightface.app.FaceAnalysis
    """
    
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    
    current_ratio = bbox_height / bbox_width
    if current_ratio > target_ratio:
        more_height = bbox_height * expand_coef
        more_width = (bbox_height+more_height) / target_ratio - bbox_width
    elif current_ratio <= target_ratio:
        more_width = bbox_width * expand_coef
        more_height = (bbox_width+more_width) * target_ratio - bbox_height
    
    bbox_new = [0,0,0,0]
    bbox_new[0] = int(round(bbox[0] - more_width*0.5))
    bbox_new[2] = int(round(bbox[2] + more_width*0.5))
    bbox_new[1] = int(round(bbox[1] - more_height*0.5))
    bbox_new[3] = int(round(bbox[3] + more_height*0.5))
    return bbox_new

def crop_face(img_tensor, bbox_new, target_size, fill_value):
    """
    img_tensor: [3,H,W]
    bbox_new: [width_small, height_small, width_large, height_large]
    target_size: [width,height]
    fill_value: value used if need to pad
    """
    img_height, img_width = img_tensor.shape[-2:]
    
    idx_left = max(bbox_new[0],0)
    idx_right = min(bbox_new[2], img_width)
    idx_bottom = max(bbox_new[1],0)
    idx_top = min(bbox_new[3], img_height)

    pad_left = max(-bbox_new[0],0)
    pad_right = max(-(img_width-bbox_new[2]),0)
    pad_top = max(-bbox_new[1],0)
    pad_bottom = max(-(img_height-bbox_new[3]),0)

    img_face = img_tensor[:,idx_bottom:idx_top,idx_left:idx_right]
    if pad_left>0 or pad_top>0 or pad_right>0 or pad_bottom>0:
        img_face = torchvision.transforms.Pad([pad_left,pad_top,pad_right,pad_bottom], fill=fill_value)(img_face)
    img_face = torchvision.transforms.Resize(size=target_size)(img_face)
    return img_face

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


class PromptsDataset(Dataset):
    def __init__(
        self,
        prompts,
    ):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, i):
        return self.prompts[i]


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

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
    else:
        raise ValueError("--report_to must be set to 'wanb', others are not implemented.")
    
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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    now = datetime.now(my_timezone)
    timestring = f"{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}"
    folder_name = f"BS-{args.train_images_per_prompt_GPU*accelerator.num_processes}_wImg-{args.weight_loss_img}-{args.factor1}-{args.factor2}_wFace-{args.weight_loss_face}_Th-{args.uncertainty_threshold}_loraR-{args.rank}_lr-{args.learning_rate}_{timestring}"
    
    args.imgs_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "imgs")
    args.ckpts_save_dir = os.path.join(args.output_dir, args.proj_name, folder_name, "ckpts")

    if accelerator.is_main_process:
        os.makedirs(args.imgs_save_dir, exist_ok=True)
        os.makedirs(args.ckpts_save_dir, exist_ok=True)
        accelerator.init_trackers(
            args.proj_name, 
            init_kwargs = {
                "wandb": {
                    "name": folder_name, 
                    "dir": args.output_dir
                        }
                }
            )

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
    

    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    vae.enable_gradient_checkpointing()

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
    vae.to(accelerator.device, dtype=weight_dtype)

    # create a Stable DiffusionPipeline to use some of its methods
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
    
    # if args.enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         import xformers

    #         xformers_version = version.parse(xformers.__version__)
    #         if xformers_version == version.parse("0.0.16"):
    #             logger.warn(
    #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
    #             )
    #         unet.enable_xformers_memory_efficient_attention()
    #         vae.enable_xformers_memory_efficient_attention()

    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    prefix_tokens = [f"<common-token{i+1}>" for i in range(args.train_num_tokens)]
    prefix_tokens_id = expand_tokenizer(tokenizer, text_encoder, prefix_tokens)
    print(f"before, {accelerator.device}, {text_encoder.get_input_embeddings().weight.data[-1,-1]}")
    torch.distributed.broadcast(text_encoder.get_input_embeddings().weight.data, src=0)
    print(f"after, {accelerator.device}, {text_encoder.get_input_embeddings().weight.data[-1,-1]}")

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

    # import pdb; pdb.set_trace()

    
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

    # Dataset and DataLoaders creation:
    with open(args.prompt_occupation_path, 'r') as f:
        experiment_data = json.load(f)
    prompts_train = [prompt.format(occupation=occupation) for prompt in experiment_data["prompt_templates_train"] for occupation in experiment_data["occupations_train_set"]]
    
    train_dataset = PromptsDataset(prompts=prompts_train)
    args.num_update_steps_per_epoch = train_dataset.__len__()
    args.num_train_epochs = math.ceil(args.max_train_steps / args.num_update_steps_per_epoch)

    # self-make a simple dataloader
    # the created train_dataloader_idxs should be identical across devices
    random.seed(args.seed+1)
    train_dataloader_idxs = []
    for epoch in range(args.num_train_epochs):
        idxs = list(range(train_dataset.__len__()))
        random.shuffle(idxs)
        train_dataloader_idxs.append(idxs)

    
    prompts_val = [prompt.format(occupation=occupation) for prompt in experiment_data["prompt_templates_test"] for occupation in experiment_data["occupations_val_set"]]
    
    
    #######################################################
    # set up things needed for finetuning        
    gender_classifier = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT, width_mult=1.0, reduced_tail=False, dilated=False)
    gender_classifier._modules['classifier'][3] = nn.Linear(1280, 80, bias=True)
    
    gender_classifier.load_state_dict(torch.load(args.classifier_weight_path))
    gender_classifier.to(accelerator.device, dtype=weight_dtype)
    gender_classifier.requires_grad_(False)
    gender_classifier.eval()
    
    # set up face_recognition and face_app on all devices
    import face_recognition
    face_app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=['detection'], 
        providers=['CUDAExecutionProvider'], 
        provider_options=[{'device_id': accelerator.device.index}]
        )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    
    clip_image_processoor = CLIPImageProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
    clip_vision_model_w_proj = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
    clip_vision_model_w_proj.vision_model.to(accelerator.device, dtype=weight_dtype)
    clip_vision_model_w_proj.visual_projection.to(accelerator.device, dtype=weight_dtype)
    clip_vision_model_w_proj.requires_grad_(False)
    clip_vision_model_w_proj.gradient_checkpointing_enable()
    clip_img_mean = torch.tensor(clip_image_processoor.image_mean).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype) # mean is based on range [0,1]
    clip_img_std = torch.tensor(clip_image_processoor.image_std).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype) # std is based on range [0,1]
    

    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2.to(accelerator.device, dtype=weight_dtype)
    dinov2.requires_grad_(False)
    dinov2_img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)
    dinov2_img_std = torch.tensor([0.229, 0.224, 0.225]).reshape([-1,1,1]).to(accelerator.device, dtype=weight_dtype)
    
    CE_loss = nn.CrossEntropyLoss(reduction="none")   

    # build opensphere model
    sys.path.append(Path(__file__).parent.parent.__str__())
    sys.path.append(Path(__file__).parent.parent.joinpath("opensphere").__str__())
    from opensphere.builder import build_from_cfg
    from opensphere.utils import fill_config

    with open(args.opensphere_config, 'r') as f:
        opensphere_config = yaml.load(f, yaml.SafeLoader)
    opensphere_config['data'] = fill_config(opensphere_config['data'])
    face_feats_net = build_from_cfg(
        opensphere_config['model']['backbone']['net'],
        'model.backbone',
    )
    face_feats_net = nn.DataParallel(face_feats_net)
    face_feats_net.load_state_dict(torch.load(args.opensphere_model_path))
    face_feats_net = face_feats_net.module
    face_feats_net.to(accelerator.device)
    face_feats_net.requires_grad_(False)
    face_feats_net.to(weight_dtype)
    face_feats_net.eval()
    
    face_feats_model = FaceFeatsModel(args.face_feats_path)
    face_feats_model.to(weight_dtype_high_precision)
    face_feats_model.to(accelerator.device)
    face_feats_model.eval()

    #######################################################
    
    @torch.no_grad()
    def generate_image_no_gradient(prompt, noises, num_denoising_steps, which_text_encoder, which_unet, which_prefix_embedding):
        """
        prompts: str
        noises: [N,4,64,64], N is number images to be generated for the prompt
        """
        N = noises.shape[0]
        prompts = [prompt] * N
        
        prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
        prompts_token["input_ids"] = prompts_token["input_ids"].to(accelerator.device)
        prompts_token["attention_mask"] = prompts_token["attention_mask"].to(accelerator.device)

        if which_prefix_embedding != None:
            # encoder_hidden_states = self.text_encoder(prompts_token["input_ids"], prompts_token["attention_mask"])[0]
            # the following code segment does exactly the same thing as the above line
            token_embeds = which_text_encoder.text_model.embeddings(
                input_ids=prompts_token["input_ids"],
                position_ids=None
                )
            token_embeds = which_prefix_embedding(
                input_ids=prompts_token["input_ids"],
                position_ids=None,
                unfair_embeds=token_embeds,
            )
            # token_embeds = token_embeds.to(weight_dtype)
            encoder_hidden_states = text_model_forward(
                which_text_encoder,
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
        else:
            prompt_embeds = which_text_encoder(
                prompts_token["input_ids"],
                prompts_token["attention_mask"],
            )
            which_text_encoder(prompts_token["input_ids"],prompts_token["attention_mask"],)
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
            uncond_input["input_ids"] = uncond_input["input_ids"].to(accelerator.device)
            uncond_input["attention_mask"] = uncond_input["attention_mask"].to(accelerator.device)
            negative_prompt_embeds = which_text_encoder(
                uncond_input["input_ids"],
                None,
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
            
            noises_pred = which_unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(weight_dtype_high_precision)
            
            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + args.guidance_scale * (noises_pred_text - noises_pred_uncond)
            
            latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1,1) # in range [-1,1]
        
        return images
    
    def generate_image_w_gradient(prompt, noises, num_denoising_steps, which_text_encoder, which_unet, which_prefix_embedding):
        """
        prompts: str
        noises: [N,4,64,64], N is number images to be generated for the prompt
        """
        # to enable gradient_checkpointing, unet must be set to train()
        unet.train()
        
        N = noises.shape[0]
        prompts = [prompt] * N
        
        prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
        prompts_token["input_ids"] = prompts_token["input_ids"].to(accelerator.device)
        prompts_token["attention_mask"] = prompts_token["attention_mask"].to(accelerator.device)

        if which_prefix_embedding != None:
            # encoder_hidden_states = self.text_encoder(prompts_token["input_ids"], prompts_token["attention_mask"])[0]
            # the following code segment does exactly the same thing as the above line
            token_embeds = which_text_encoder.text_model.embeddings(
                input_ids=prompts_token["input_ids"],
                position_ids=None
                )
            token_embeds = which_prefix_embedding(
                input_ids=prompts_token["input_ids"],
                position_ids=None,
                unfair_embeds=token_embeds,
            )
            # token_embeds = token_embeds.to(weight_dtype)
            encoder_hidden_states = text_model_forward(
                which_text_encoder,
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
        else:
            prompt_embeds = which_text_encoder(
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
            uncond_input["input_ids"] = uncond_input["input_ids"].to(accelerator.device)
            uncond_input["attention_mask"] = uncond_input["attention_mask"].to(accelerator.device)
            negative_prompt_embeds = which_text_encoder(
                uncond_input["input_ids"],
                None,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
        prompt_embeds = prompt_embeds.to(weight_dtype)
        
        noise_scheduler.set_timesteps(num_denoising_steps)
        grad_coefs = []
        for i, t in enumerate(noise_scheduler.timesteps):
            grad_coefs.append( noise_scheduler.alphas_cumprod[t].sqrt().item() * (1-noise_scheduler.alphas_cumprod[t]).sqrt().item() / (1-noise_scheduler.alphas[t].item()) )
        grad_coefs = np.array(grad_coefs)
        grad_coefs /= (math.prod(grad_coefs)**(1/len(grad_coefs)))
            
        latents = noises
        for i, t in enumerate(noise_scheduler.timesteps):
        
            # scale model input
            latent_model_input = torch.cat([latents.detach().to(weight_dtype)]*2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            
            noises_pred = which_unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            noises_pred = noises_pred.to(weight_dtype_high_precision)
            
            noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
            noises_pred = noises_pred_uncond + args.guidance_scale * (noises_pred_text - noises_pred_uncond)
            
            hook_fn = make_grad_hook(grad_coefs[i])
            noises_pred.register_hook(hook_fn)
            
            latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1,1) # in range [-1,1]
        
        return images
    
    
    def get_clip_feat(images, normalize=True, to_high_precision=True):
        """get clip features

        Args:
            images (torch.tensor): shape [N,3,H,W], in range [-1,1]
            normalize (bool):
            to_high_precision (bool):

        Returns:
            embeds (torch.tensor)
        """
        images_preprocessed = ((images+1)*0.5 - clip_img_mean) / clip_img_std
        embeds = clip_vision_model_w_proj(images_preprocessed).image_embeds
        
        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds
    
    def get_dino_feat(images, normalize=True, to_high_precision=True):
        """get dino features

        Args:
            images (torch.tensor): shape [N,3,H,W], in range [-1,1]
            normalize (bool):
            to_high_precision (bool):

        Returns:
            embeds (torch.tensor)
        """
        images_preprocessed = ((images+1)*0.5 - dinov2_img_mean) / dinov2_img_std
        embeds = dinov2(images_preprocessed)
        
        if to_high_precision:
            embeds = embeds.to(torch.float)
        if normalize:
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
        return embeds
    
    def get_face_feats(net, data, flip=True, normalize=True, to_high_precision=True):
        # extract features from the original 
        # and horizontally flipped data
        feats = net(data)
        if flip:
            data = torch.flip(data, [3])
            feats += net(data)
        if to_high_precision:
            feats = feats.to(torch.float)
        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats
    
    def image_pipeline(img, tgz_landmark):
        img = (img+1)/2.0 * 255 # map to [0,255]

        crop_size = (112,112)
        src_landmark = np.array(
        [[38.2946, 51.6963], # left eye
        [73.5318, 51.5014], # right eye
        [56.0252, 71.7366], # nose
        [41.5493, 92.3655], # left corner of the mouth
        [70.7299, 92.2041]] # right corner of the mouth
        )

        tform = transform.SimilarityTransform()
        tform.estimate(tgz_landmark, src_landmark)

        M = torch.tensor(tform.params[0:2, :]).unsqueeze(dim=0).to(img.dtype).to(img.device)
        img_face = kornia.geometry.transform.warp_affine(img.unsqueeze(dim=0), M, crop_size, mode='bilinear', padding_mode='zeros', align_corners=False)
        img_face = img_face.squeeze()

        img_face = (img_face/255.0)*2-1 # map back to [-1,1]
        return img_face
        
    def get_face(images, fill_value=-1):
        """
        images:shape [N,3,H,W], in range [-1,1], pytorch tensor
        returns:
            face_indicators: torch tensor of shape [N], only True or False
                True means face is detected, False otherwise
            face_bboxs: torch tensor of shape [N,4], 
                if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
            face_chips: torch tensor of shape [N,3,224,224]
                if face_indicator is False, the corresponding face_chip will be all fill_value
        """
        face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app = get_face_app(images, fill_value=fill_value)

        if face_indicators_app.logical_not().sum() > 0:
            face_indicators_FR, face_bboxs_FR, face_chips_FR, face_landmarks_FR, aligned_face_chips_FR = get_face_FR(images[face_indicators_app.logical_not()], fill_value=fill_value)

            face_bboxs_app[face_indicators_app.logical_not()] = face_bboxs_FR
            face_chips_app[face_indicators_app.logical_not()] = face_chips_FR
            face_landmarks_app[face_indicators_app.logical_not()] = face_landmarks_FR
            aligned_face_chips_app[face_indicators_app.logical_not()] = aligned_face_chips_FR

            face_indicators_app[face_indicators_app.logical_not()] = face_indicators_FR

        return face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app

    
    def get_largest_face_FR(faces_from_FR, dim_max, dim_min):
        if len(faces_from_FR) == 1:
            return faces_from_FR[0]
        elif len(faces_from_FR) > 1:
            area_max = 0
            idx_max = 0
            for idx, bbox in enumerate(faces_from_FR):
                bbox1 = np.array((bbox[-1],) + bbox[:-1])
                area = (min(bbox1[2],dim_max) - max(bbox1[0], dim_min)) * (min(bbox1[3],dim_max) - max(bbox1[1], dim_min))
                if area > area_max:
                    area_max = area
                    idx_max = idx
            return faces_from_FR[idx_max]
    
    def get_face_FR(images, fill_value=-1):
        """
        images:shape [N,3,H,W], in range [-1,1], pytorch tensor
        returns:
            face_indicators: torch tensor of shape [N], only True or False
                True means face is detected, False otherwise
            face_bboxs: torch tensor of shape [N,4], 
                if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
            face_chips: torch tensor of shape [N,3,224,224]
                if face_indicator is False, the corresponding face_chip will be all fill_value
        """

        images_np = ((images*0.5 + 0.5)*255).cpu().detach().permute(0,2,3,1).float().numpy().astype(np.uint8)
        
        face_indicators_FR = []
        face_bboxs_FR = []
        face_chips_FR = []
        face_landmarks_FR = []
        aligned_face_chips_FR = []
        for idx, image_np in enumerate(images_np):
            faces_from_FR = face_recognition.face_locations(image_np, model="cnn", number_of_times_to_upsample=0)
            if len(faces_from_FR) == 0:
                face_indicators_FR.append(False)
                face_bboxs_FR.append([fill_value]*4)
                face_chips_FR.append(torch.ones([1,3,args.size_face,args.size_face], dtype=images.dtype, device=images.device)*(fill_value))
                face_landmarks_FR.append(torch.ones([1,5,2], dtype=images.dtype, device=images.device)*(fill_value))
                aligned_face_chips_FR.append(torch.ones([1,3,args.size_aligned_face,args.size_aligned_face], dtype=images.dtype, device=images.device)*(fill_value))
            else:
                face_from_FR = get_largest_face_FR(faces_from_FR, dim_max=image_np.shape[0], dim_min=0)
                bbox = face_from_FR
                bbox = np.array((bbox[-1],) + bbox[:-1]) # need to convert bbox from face_recognition to the right order
                bbox = expand_bbox(bbox, expand_coef=1.1, target_ratio=1) # need to use a larger expand_coef for FR
                face_chip = crop_face(images[idx], bbox, target_size=[args.size_face,args.size_face], fill_value=fill_value)
                
                face_landmarks = face_recognition.face_landmarks(image_np, face_locations=[face_from_FR], model="large")

                left_eye = np.array(face_landmarks[0]["left_eye"]).mean(axis=0)
                right_eye = np.array(face_landmarks[0]["right_eye"]).mean(axis=0)
                nose_tip = np.array(face_landmarks[0]["nose_bridge"][-1])
                top_lip_left = np.array(face_landmarks[0]["top_lip"][0])
                top_lip_right = np.array(face_landmarks[0]["top_lip"][6])
                face_landmarks = np.stack([left_eye, right_eye, nose_tip, top_lip_left, top_lip_right])
                
                aligned_face_chip = image_pipeline(images[idx], face_landmarks)
                
                face_indicators_FR.append(True)
                face_bboxs_FR.append(bbox)
                face_chips_FR.append(face_chip.unsqueeze(dim=0))
                face_landmarks_FR.append(torch.tensor(face_landmarks).unsqueeze(dim=0).to(device=images.device).to(images.dtype))
                aligned_face_chips_FR.append(aligned_face_chip.unsqueeze(dim=0))
        
        face_indicators_FR = torch.tensor(face_indicators_FR).to(device=images.device)
        face_bboxs_FR = torch.tensor(face_bboxs_FR).to(device=images.device)
        face_chips_FR = torch.cat(face_chips_FR, dim=0)
        face_landmarks_FR = torch.cat(face_landmarks_FR, dim=0)
        aligned_face_chips_FR = torch.cat(aligned_face_chips_FR, dim=0)
        
        return face_indicators_FR, face_bboxs_FR, face_chips_FR, face_landmarks_FR, aligned_face_chips_FR

    def get_largest_face_app(face_from_app, dim_max, dim_min):
        if len(face_from_app) == 1:
            return face_from_app[0]
        elif len(face_from_app) > 1:
            area_max = 0
            idx_max = 0
            for idx in range(len(face_from_app)):
                bbox = face_from_app[idx]["bbox"]
                area = (min(bbox[2],dim_max) - max(bbox[0], dim_min)) * (min(bbox[3],dim_max) - max(bbox[1], dim_min))
                if area > area_max:
                    area_max = area
                    idx_max = idx
            return face_from_app[idx_max]
    
    def get_face_app(images, fill_value=-1):
        """
        images:shape [N,3,H,W], in range [-1,1], pytorch tensor
        returns:
            face_indicators: torch tensor of shape [N], only True or False
                True means face is detected, False otherwise
            face_bboxs: torch tensor of shape [N,4], 
                if face_indicator is False, the corresponding face_bbox will be [fill_value,fill_value,fill_value,fill_value]
            face_chips: torch tensor of shape [N,3,224,224]
                if face_indicator is False, the corresponding face_chip will be all fill_value
        """        
        images_np = ((images*0.5 + 0.5)*255).cpu().detach().permute(0,2,3,1).float().numpy().astype(np.uint8)
        
        face_indicators_app = []
        face_bboxs_app = []
        face_chips_app = []
        face_landmarks_app = []
        aligned_face_chips_app = []
        for idx, image_np in enumerate(images_np):
            # face_app.get input should be [BGR]
            faces_from_app = face_app.get(image_np[:,:,[2,1,0]])
            if len(faces_from_app) == 0:
                face_indicators_app.append(False)
                face_bboxs_app.append([fill_value]*4)
                face_chips_app.append(torch.ones([1,3,args.size_face,args.size_face], dtype=images.dtype, device=images.device)*(fill_value))
                face_landmarks_app.append(torch.ones([1,5,2], dtype=images.dtype, device=images.device)*(fill_value))
                aligned_face_chips_app.append(torch.ones([1,3,args.size_aligned_face,args.size_aligned_face], dtype=images.dtype, device=images.device)*(fill_value))
            else:
                face_from_app = get_largest_face_app(faces_from_app, dim_max=image_np.shape[0], dim_min=0)
                bbox = expand_bbox(face_from_app["bbox"], expand_coef=0.5, target_ratio=1)
                face_chip = crop_face(images[idx], bbox, target_size=[args.size_face,args.size_face], fill_value=fill_value)
                
                face_landmarks = np.array(face_from_app["kps"])
                aligned_face_chip = image_pipeline(images[idx], face_landmarks)
                
                face_indicators_app.append(True)
                face_bboxs_app.append(bbox)
                face_chips_app.append(face_chip.unsqueeze(dim=0))
                face_landmarks_app.append(torch.tensor(face_landmarks).unsqueeze(dim=0).to(device=images.device).to(images.dtype))
                aligned_face_chips_app.append(aligned_face_chip.unsqueeze(dim=0))
        
        face_indicators_app = torch.tensor(face_indicators_app).to(device=images.device)
        face_bboxs_app = torch.tensor(face_bboxs_app).to(device=images.device)
        face_chips_app = torch.cat(face_chips_app, dim=0)
        face_landmarks_app = torch.cat(face_landmarks_app, dim=0)
        aligned_face_chips_app = torch.cat(aligned_face_chips_app, dim=0)
        
        return face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app
                
    def get_face_gender(face_chips, selector=None, fill_value=-1):
        """for CelebA classifier
        """
        if selector != None:
            face_chips_w_faces = face_chips[selector]
        else:
            face_chips_w_faces = face_chips
            
        if face_chips_w_faces.shape[0] == 0:
            logits_gender = torch.empty([0,2], dtype=face_chips.dtype, device=face_chips.device)
            probs_gender = torch.empty([0,2], dtype=face_chips.dtype, device=face_chips.device)
            # pred_class_probs_gender = torch.empty([0], dtype=face_chips.dtype, device=face_chips.device)
            preds_gender = torch.empty([0], dtype=torch.int64, device=face_chips.device)
        else:
            logits = gender_classifier(face_chips_w_faces)
            logits_gender = logits.view([logits.shape[0],-1,2])[:,20,:]
            probs_gender = torch.softmax(logits_gender, dim=-1)
        
            temp = probs_gender.max(dim=-1)
            # pred_class_probs_gender = temp.values
            preds_gender = temp.indices
        
        if selector != None:
            preds_gender_new = torch.ones(
                [selector.shape[0]]+list(preds_gender.shape[1:]), 
                dtype=preds_gender.dtype, 
                device=preds_gender.device
                ) * (fill_value)
            preds_gender_new[selector] = preds_gender
            
            probs_gender_new = torch.ones(
                [selector.shape[0]]+list(probs_gender.shape[1:]),
                dtype=probs_gender.dtype, 
                device=probs_gender.device
                ) * (fill_value)
            probs_gender_new[selector] = probs_gender
            
            logits_gender_new = torch.ones(
                [selector.shape[0]]+list(logits_gender.shape[1:]),
                dtype=logits_gender.dtype, 
                device=logits_gender.device
                ) * (fill_value)
            logits_gender_new[selector] = logits_gender
            
            return preds_gender_new, probs_gender_new, logits_gender_new
        else:
            return preds_gender, probs_gender, logits_gender
        
    @torch.no_grad()
    def generate_dynamic_targets(probs, target_ratio=0.5, w_uncertainty=False):
        """generate dynamic targets for the distributional alignment loss

        Args:
            probs (torch.tensor): shape [N,2], N points in a probability simplex of 2 dims
            target_ratio (float): target distribution, the percentage of class 1 (male)
            w_uncertainty (True/False): whether return uncertainty measures
        
        Returns:
            targets_all (torch.tensor): target classes
            uncertainty_all (torch.tensor): uncertainty of target classes
        """
        idxs_2_rank = (probs!=-1).all(dim=-1)
        probs_2_rank = probs[idxs_2_rank]

        rank = torch.argsort(torch.argsort(probs_2_rank[:,1]))
        targets = (rank >= (rank.shape[0]*target_ratio)).long()

        targets_all = torch.ones([probs.shape[0]], dtype=torch.long, device=probs.device) * (-1)
        targets_all[idxs_2_rank] = targets
        
        if w_uncertainty:
            uncertainty = torch.ones([probs_2_rank.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
            uncertainty[targets==1] = torch.tensor(
                1 - scipy.stats.binom.cdf(
                    (rank[targets==1]).cpu().numpy(), 
                    probs_2_rank.shape[0], 
                    1-target_ratio
                    )
                ).to(probs.dtype).to(probs.device)
            uncertainty[targets==0] = torch.tensor(
                scipy.stats.binom.cdf(
                    rank[targets==0].cpu().numpy(), 
                    probs_2_rank.shape[0], 
                    target_ratio
                    )
                ).to(probs.dtype).to(probs.device)
            
            uncertainty_all = torch.ones([probs.shape[0]], dtype=probs.dtype, device=probs.device) * (-1)
            uncertainty_all[idxs_2_rank] = uncertainty
            
            return targets_all, uncertainty_all
        else:
            return targets_all

    @torch.no_grad()
    def evaluate_process(which_text_encoder, which_unet, which_prefix_embedding, name, prompts, noises, current_global_step):
        logs = []
        log_imgs = []
        num_denoising_steps = 25
        for prompt_i, noises_i in itertools.zip_longest(prompts, noises):
            prompt_i_debiased = prompt_debiaser(prompt_i)
            if accelerator.is_main_process:
                logs_i = {
                    "gender_gap": [],
                    "gender_gap_abs": [],
                    "gender_pred_between_0.2_0.8": [],
                }
                log_imgs_i = {}
            ################################################
            # step 1: generate all ori images
            images_ori = []
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                images_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet, which_prefix_embedding=None)
                images_ori.append(images_ij)
            images_ori = torch.cat(images_ori)
            face_indicators_ori, face_bboxs_ori, face_chips_ori, face_landmarks_ori, aligned_face_chips_ori = get_face(images_ori)
            preds_gender_ori, probs_gender_ori, logits_gender_ori = get_face_gender(face_chips_ori, selector=face_indicators_ori, fill_value=-1)
            
            face_feats_ori = get_face_feats(face_feats_net, aligned_face_chips_ori)
            _, face_real_scores_ori = face_feats_model.semantic_search(face_feats_ori, selector=face_indicators_ori, return_similarity=True)

            images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
            face_indicators_ori_all = customized_all_gather(face_indicators_ori, accelerator, return_tensor_other_processes=False)
            face_bboxs_ori_all = customized_all_gather(face_bboxs_ori, accelerator, return_tensor_other_processes=False)
            preds_gender_ori_all = customized_all_gather(preds_gender_ori, accelerator, return_tensor_other_processes=False)
            probs_gender_ori_all = customized_all_gather(probs_gender_ori, accelerator, return_tensor_other_processes=False)
            face_real_scores_ori_all = customized_all_gather(face_real_scores_ori, accelerator, return_tensor_other_processes=False)

            if accelerator.is_main_process:
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_ori.jpg")
                plot_in_grid(
                    images_ori_all, 
                    save_to, 
                    face_indicators=face_indicators_ori_all, face_bboxs=face_bboxs_ori_all, 
                    preds_gender=preds_gender_ori_all, 
                    pred_class_probs_gender=probs_gender_ori_all.max(dim=-1).values,
                    # face_real_scores=face_real_scores_ori_all
                )

                log_imgs_i["img_ori"] = [save_to]

            
            images = []
            N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
            for j in range(N):
                noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                images_ij = generate_image_no_gradient(prompt_i_debiased, noises_ij, num_denoising_steps, which_text_encoder=which_text_encoder, which_unet=which_unet, which_prefix_embedding=which_prefix_embedding)
                images.append(images_ij)
            images = torch.cat(images)
            
            face_indicators, face_bboxs, face_chips, face_landmarks, aligned_face_chips = get_face(images)
            preds_gender, probs_gender, logits_gender = get_face_gender(face_chips, selector=face_indicators, fill_value=-1)
            
            face_feats = get_face_feats(face_feats_net, aligned_face_chips)
            _, face_real_scores = face_feats_model.semantic_search(face_feats, selector=face_indicators, return_similarity=True)

            images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
            face_indicators_all = customized_all_gather(face_indicators, accelerator, return_tensor_other_processes=False)
            face_bboxs_all = customized_all_gather(face_bboxs, accelerator, return_tensor_other_processes=False)
            preds_gender_all = customized_all_gather(preds_gender, accelerator, return_tensor_other_processes=False)
            probs_gender_all = customized_all_gather(probs_gender, accelerator, return_tensor_other_processes=False)
            face_real_scores = customized_all_gather(face_real_scores, accelerator, return_tensor_other_processes=False)

            if accelerator.is_main_process:
                save_to = os.path.join(args.imgs_save_dir, f"eval_{name}_{global_step}_{prompt_i}_generated.jpg")
                plot_in_grid(
                    images_all, 
                    save_to, 
                    face_indicators=face_indicators_all, 
                    face_bboxs=face_bboxs_all, 
                    preds_gender=preds_gender_all, 
                    pred_class_probs_gender=probs_gender_all.max(dim=-1).values,
                    # face_real_scores=face_real_scores
                    )

                log_imgs_i["img_generated"] = [save_to]
            
            if accelerator.is_main_process:
                probs_tmp = probs_gender_all[(probs_gender_all!=-1).all(dim=-1)]
                gender_gap = (((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean() - ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean()).item()
                gender_pred_between_02_08 = ((probs_tmp[:,1]>=0.2)*(probs_tmp[:,1]<=0.8)).float().mean().item()
                logs_i["gender_gap"].append(gender_gap)
                logs_i["gender_gap_abs"].append(abs(gender_gap))
                logs_i["gender_pred_between_0.2_0.8"].append(abs(gender_pred_between_02_08))

            
            if accelerator.is_main_process:
                log_imgs.append(log_imgs_i)
                logs.append(logs_i)
        
        if accelerator.is_main_process:
            for prompt_i, logs_i in itertools.zip_longest(prompts, logs):
                for key, values in logs_i.items():
                    if isinstance(values, list):
                        wandb_tracker.log({f"eval_{name}_{key}_{prompt_i}": np.mean(values)}, step=current_global_step)
                    else:
                        wandb_tracker.log({f"eval_{name}_{key}_{prompt_i}": values.mean().item()}, step=current_global_step)
                
                for key in list(logs[0].keys()):
                    avg = np.array([log[key] for log in logs]).mean()
                    wandb_tracker.log({f"eval_{name}_{key}": avg}, step=current_global_step)

            imgs_dict = {}
            for prompt_i, log_imgs_i in itertools.zip_longest(prompts, log_imgs):
                for key, values in log_imgs_i.items():
                    if key not in imgs_dict.keys():
                        imgs_dict[key] = [wandb.Image(
                            data_or_path=values[0],
                            caption=prompt_i,
                        )]
                    else:
                        imgs_dict[key].append(wandb.Image(
                            data_or_path=values[0],
                            caption=prompt_i,
                        ))
            for key, imgs in imgs_dict.items():
                wandb_tracker.log(
                    {f"eval_{name}_{key}": imgs},
                    step=current_global_step
                    ) 
        
        return logs, log_imgs
    
    def apply_grad_hook_face(images, face_bboxs, face_bboxs_ori, targets, preds_gender_ori, probs_gender_ori, factor=0.1):
        """apply gradient hook on non-face regions of the generated images
        """
        images_new = []
        for image, face_bbox, face_bbox_ori, target, pred_gender_ori, prob_gender_ori in itertools.zip_longest(images, face_bboxs, face_bboxs_ori, targets, preds_gender_ori, probs_gender_ori):
            if (face_bbox == -1).all():
                images_new.append(image.unsqueeze(dim=0))
            else:
                img_width, img_height = image.shape[1:]
                idx_left = max(face_bbox[0], face_bbox_ori[0], 0)
                idx_right = min(face_bbox[2], face_bbox_ori[2], img_width)
                idx_bottom = max(face_bbox[1], face_bbox_ori[1], 0)
                idx_top = min(face_bbox[3], face_bbox_ori[3], img_height)

                img_face = image[:,idx_bottom:idx_top,idx_left:idx_right].clone()
                if target==-1:
                    grad_hook = make_grad_hook(factor)
                elif target==pred_gender_ori:
                    grad_hook = make_grad_hook(1)
                elif target!=pred_gender_ori:
                    grad_hook = make_grad_hook(factor)
                img_face.register_hook(grad_hook)

                img_add = torch.zeros_like(image)
                img_add[:,idx_bottom:idx_top,idx_left:idx_right] = img_face

                mask = torch.zeros_like(image)
                mask[:,idx_bottom:idx_top,idx_left:idx_right] = 1

                image = mask*img_add + (1-mask)*image
                images_new.append(image.unsqueeze(dim=0))

        images_new = torch.cat(images_new)
        return images_new
    
    def gen_dynamic_weights(face_indicators, targets, preds_gender_ori, probs_gender_ori, factor=0.2):
        weights = []
        for face_indicator, target, pred_gender_ori, prob_gender_ori in itertools.zip_longest(face_indicators, targets, preds_gender_ori, probs_gender_ori):
            if (face_indicator == False).all():
                weights.append(1)
            else:
                if target==-1:
                    weights.append(factor)
                elif target==pred_gender_ori:
                    weights.append(1)
                elif target!=pred_gender_ori:
                    weights.append(factor)

        weights = torch.tensor(weights, dtype=probs_gender_ori.dtype, device=probs_gender_ori.device)
        return weights

    def model_sanity_print(model, state):
        params = [p for p in model.parameters()]
        print(f"\t{accelerator.device}; {state};\n\t\tparam[0]: {params[0].flatten()[-1].item():.8f};\tparam[0].grad: {params[0].grad.flatten()[-1].item():.8f};")


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
        
    def evaluation_step(current_step):
        noises_val = torch.randn(
        [len(prompts_val), args.val_images_per_prompt_GPU,4,64,64],
        dtype=weight_dtype_high_precision
        ).to(accelerator.device)
        evaluate_process(text_encoder, unet, prefix_embedding, "main", prompts_val, noises_val, current_step)

        # evaluate EMA as well
        with torch.no_grad():
            prefix_embedding_copy = copy.deepcopy(prefix_embedding)
            for p, p_from in itertools.zip_longest(list(prefix_embedding.token_embedding.parameters()), prefix_embedding_ema.shadow_params):
                p.data = p_from.data
            
        evaluate_process(text_encoder, unet, prefix_embedding, "EMA", prompts_val, noises_val, current_step)
        
        with torch.no_grad():
            for p, p_from in itertools.zip_longest(list(prefix_embedding.token_embedding.parameters()), list(prefix_embedding_copy.token_embedding.parameters())):
                p.data = p_from.data
    
    
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num prompts = {train_dataset.__len__()}")
    logger.info(f"  Num images per prompt = {args.train_images_per_prompt_GPU} (per GPU) * {accelerator.num_processes} (GPU)")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:

        if not os.path.exists(args.resume_from_checkpoint):
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])

            resume_global_step = global_step
            first_epoch = global_step // args.num_update_steps_per_epoch
            resume_step = resume_global_step % (args.num_update_steps_per_epoch)
            
            prefix_embedding_ema.to(accelerator.device)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, data_idx in enumerate(train_dataloader_idxs[epoch]):            
            
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                progress_bar.update(1)
                continue

            # get prompt, should be identical across processes
            prompt_i = train_dataset.__getitem__(data_idx)
            prompt_i_debiased = prompt_debiaser(prompt_i)

            if global_step == 0:
                evaluation_step(global_step)
            
            # generate noises, should differ by processes
            noises_i = torch.randn(
                [args.train_images_per_prompt_GPU,4,64,64],
                dtype=weight_dtype_high_precision
                ).to(accelerator.device)

            accelerator.wait_for_everyone()
            optimizer.zero_grad()
            # logs = []
            # log_imgs = []

            # print noise to check if they are different by device
            noises_i_all = [noises_i.detach().clone() for i in range(accelerator.num_processes)]
            torch.distributed.all_gather(noises_i_all, noises_i)
            if accelerator.is_main_process:
                now = datetime.now(my_timezone)
                accelerator.print(
                    f"{now.strftime('%Y/%m/%d - %H:%M:%S')} --- epoch: {epoch}, step: {step}, prompt: {prompt_i}\n" +
                    " ".join([f"\tprocess idx: {idx}; noise: {noises_i_all[idx].flatten()[-1].item():.4f};" for idx in range(len(noises_i_all))])
                    )
            
            if accelerator.is_main_process:
                logs_i = {
                    "loss_fair": [],
                    "loss_face": [],
                    "loss_CLIP": [],
                    "loss_DINO": [],
                    "loss": [],
                    "gender_gap": [],
                    "gender_gap_abs": [],
                    "gender_pred_between_0.2_0.8": [],
                }
                log_imgs_i = {}

            num_denoising_steps = random.choices(range(19,24), k=1)
            torch.distributed.broadcast_object_list(num_denoising_steps, src=0)
            num_denoising_steps = num_denoising_steps[0]

            with torch.no_grad():
                ################################################
                # step 1: generate all images using the diffusion model being finetuned
                images = []
                N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
                for j in range(N):
                    noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                    images_ij = generate_image_no_gradient(prompt_i_debiased, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet, which_prefix_embedding=prefix_embedding)
                    images.append(images_ij)
                images = torch.cat(images)

                face_indicators, face_bboxs, face_chips, face_landmarks, aligned_face_chips = get_face(images)
                preds_gender, probs_gender, logits_gender = get_face_gender(face_chips, selector=face_indicators, fill_value=-1)

                
                face_feats = torch.ones([aligned_face_chips.shape[0],512], dtype=weight_dtype_high_precision, device=aligned_face_chips.device) * (-1)
                if sum(face_indicators)>0:
                    face_feats_ = get_face_feats(face_feats_net, aligned_face_chips[face_indicators])
                    face_feats[face_indicators] = face_feats_

                _, face_real_scores = face_feats_model.semantic_search(face_feats, selector=face_indicators, return_similarity=True)

                face_indicators_all, face_indicators_others = customized_all_gather(face_indicators, accelerator, return_tensor_other_processes=True)
                accelerator.print(f"\tNum faces detected: {face_indicators_all.sum().item()}/{face_indicators_all.shape[0]}.")
                
                images_all = customized_all_gather(images, accelerator, return_tensor_other_processes=False)
                face_bboxs_all = customized_all_gather(face_bboxs, accelerator, return_tensor_other_processes=False)
                preds_gender_all = customized_all_gather(preds_gender, accelerator, return_tensor_other_processes=False)
                probs_gender_all = customized_all_gather(probs_gender, accelerator, return_tensor_other_processes=False)
                face_real_scores_all = customized_all_gather(face_real_scores, accelerator, return_tensor_other_processes=False)
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_generated.jpg")
                        plot_in_grid(images_all, save_to, face_indicators=face_indicators_all, face_bboxs=face_bboxs_all, preds_gender=preds_gender_all, pred_class_probs_gender=probs_gender_all.max(dim=-1).values)

                        log_imgs_i["img_generated"] = [save_to]

                probs_gender_all, probs_gender_others = customized_all_gather(probs_gender, accelerator, return_tensor_other_processes=True)
                if accelerator.is_main_process:
                    probs_tmp = probs_gender_all[(probs_gender_all!=-1).all(dim=-1)]
                    gender_gap = (((probs_tmp[:,1]>=0.5)*(probs_tmp[:,1]<=1)).float().mean() - ((probs_tmp[:,1]>=0)*(probs_tmp[:,1]<=0.5)).float().mean()).item()
                    gender_pred_between_02_08 = ((probs_tmp[:,1]>=0.2)*(probs_tmp[:,1]<=0.8)).float().mean().item()
                    logs_i["gender_gap"].append(gender_gap)
                    logs_i["gender_gap_abs"].append(abs(gender_gap))
                    logs_i["gender_pred_between_0.2_0.8"].append(gender_pred_between_02_08)

                ################################################
                # Step 2: generate dynamic targets 
                # also broadcast from process idx 0, just in case targets_all computed might be different on different processes
                targets_all, uncertainty_all = generate_dynamic_targets(probs_gender_all, w_uncertainty=True)
                torch.distributed.broadcast(targets_all, src=0)
                torch.distributed.broadcast(uncertainty_all, src=0)

                targets_all[uncertainty_all>args.uncertainty_threshold] = -1
                targets = targets_all[probs_gender.shape[0]*(accelerator.local_process_index):probs_gender.shape[0]*(accelerator.local_process_index+1)]
                uncertainty = uncertainty_all[probs_gender.shape[0]*(accelerator.local_process_index):probs_gender.shape[0]*(accelerator.local_process_index+1)]
                accelerator.print(f"\tNum faces to compute grads: {(targets_all!=-1).sum().item()}/{targets_all.shape[0]}")

                ################################################
                # Step 3: generate all original images using the original diffusion model
                # note that only targets from above will be used to compute loss
                # all other variables will not be used below
                images_ori = []
                N = math.ceil(noises_i.shape[0] / args.val_GPU_batch_size)
                for j in range(N):
                    noises_ij = noises_i[args.val_GPU_batch_size*j:args.val_GPU_batch_size*(j+1)]
                    images_ij = generate_image_no_gradient(prompt_i, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet, which_prefix_embedding=None)
                    images_ori.append(images_ij)
                images_ori = torch.cat(images_ori)

                face_indicators_ori, face_bboxs_ori, face_chips_ori, face_landmarks_ori, aligned_face_chips_ori = get_face(images_ori)
                preds_gender_ori, probs_gender_ori, logits_gender_ori = get_face_gender(face_chips_ori, selector=face_indicators_ori, fill_value=-1)
                
                images_small_ori = transforms.Resize(args.img_size_small)(images_ori)
                clip_feats_ori = get_clip_feat(images_small_ori, normalize=True, to_high_precision=True)
                DINO_feats_ori = get_dino_feat(images_small_ori, normalize=True, to_high_precision=True)

                images_ori_all = customized_all_gather(images_ori, accelerator, return_tensor_other_processes=False)
                face_indicators_ori_all = customized_all_gather(face_indicators_ori, accelerator, return_tensor_other_processes=False)
                face_bboxs_ori_all = customized_all_gather(face_bboxs_ori, accelerator, return_tensor_other_processes=False)
                preds_gender_ori_all = customized_all_gather(preds_gender_ori, accelerator, return_tensor_other_processes=False)
                probs_gender_ori_all = customized_all_gather(probs_gender_ori, accelerator, return_tensor_other_processes=False)

                face_feats_ori = get_face_feats(face_feats_net, aligned_face_chips_ori)
                
                if accelerator.is_main_process:
                    if step % args.train_plot_every_n_iter == 0:
                        save_to = os.path.join(args.imgs_save_dir, f"train-{global_step}_ori.jpg")
                        plot_in_grid(images_ori_all, save_to, face_indicators=face_indicators_ori_all, face_bboxs=face_bboxs_ori_all, preds_gender=preds_gender_ori_all, pred_class_probs_gender=probs_gender_ori_all.max(dim=-1).values)

                        log_imgs_i["img_ori"] = [save_to]
            
            ################################################
            # Step 4: compute loss
            loss_fair_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_face_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_CLIP_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_DINO_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            loss_i = torch.ones(targets.shape, dtype=weight_dtype, device=accelerator.device) *(-1)
            
            idxs_i = list(range(targets.shape[0]))
            N_backward = math.ceil(targets.shape[0] / args.train_GPU_batch_size)
            for j in range(N_backward):
                idxs_ij = idxs_i[j*args.train_GPU_batch_size:(j+1)*args.train_GPU_batch_size]
                noises_ij = noises_i[idxs_ij]
                targets_ij = targets[idxs_ij]
                clip_feats_ori_ij = clip_feats_ori[idxs_ij]
                DINO_feats_ori_ij = DINO_feats_ori[idxs_ij]
                preds_gender_ori_ij = preds_gender_ori[idxs_ij]
                probs_gender_ori_ij = probs_gender_ori[idxs_ij]
                face_bboxs_ori_ij = face_bboxs_ori[idxs_ij]
                face_feats_ori_ij = face_feats_ori[idxs_ij]
                
                images_ij = generate_image_w_gradient(prompt_i_debiased, noises_ij, num_denoising_steps, which_text_encoder=text_encoder, which_unet=unet, which_prefix_embedding=prefix_embedding)
                face_indicators_ij, face_bboxs_ij, face_chips_ij, face_landmarks_ij, aligned_face_chips_ij = get_face(images_ij)
                preds_gender_ij, probs_gender_ij, logits_gender_ij = get_face_gender(face_chips_ij, selector=face_indicators_ij, fill_value=-1)
                
                images_ij = apply_grad_hook_face(images_ij, face_bboxs_ij, face_bboxs_ori_ij, targets_ij, preds_gender_ori_ij, probs_gender_ori_ij, factor=args.factor2)
                images_small_ij = transforms.Resize(args.img_size_small)(images_ij)
                clip_feats_ij = get_clip_feat(images_small_ij, normalize=True, to_high_precision=True)
                DINO_feats_ij = get_dino_feat(images_small_ij, normalize=True, to_high_precision=True)

                loss_CLIP_ij = - (clip_feats_ij * clip_feats_ori_ij).sum(dim=-1) + 1
                loss_DINO_ij = - (DINO_feats_ij * DINO_feats_ori_ij).sum(dim=-1) + 1
                
                loss_fair_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=accelerator.device) *(-1)
                idxs_w_face_loss = ((face_indicators_ij == True) * (targets_ij != -1)).nonzero().view([-1])
                loss_fair_ij_w_face_loss = CE_loss(logits_gender_ij[idxs_w_face_loss], targets_ij[idxs_w_face_loss])
                loss_fair_ij[idxs_w_face_loss] = loss_fair_ij_w_face_loss
                
                loss_face_ij = torch.ones(len(idxs_ij), dtype=weight_dtype, device=accelerator.device) *(-1)

                idxs_w_face_feats_from_ori = ((face_indicators_ij==True) * (targets_ij!=-1) * (targets_ij==preds_gender_ori_ij) * (probs_gender_ori_ij.max(dim=-1).values>=args.face_gender_confidence_level)).nonzero().view([-1]).tolist()
                if len(idxs_w_face_feats_from_ori)>0:
                    face_feats_1 = get_face_feats(face_feats_net, aligned_face_chips_ij[idxs_w_face_feats_from_ori])
                    face_feats_target_1 = face_feats_ori_ij[idxs_w_face_feats_from_ori]
                    loss_face_ij[idxs_w_face_feats_from_ori] = (1 - (face_feats_1*face_feats_target_1).sum(dim=-1)).to(loss_face_ij.dtype)
                
                idxs_w_face_feats_from_search = list(set(((face_indicators_ij==True) * (targets_ij!=-1) ).nonzero().view([-1]).tolist()) - set(idxs_w_face_feats_from_ori))
                if len(idxs_w_face_feats_from_search)>0:
                    face_feats_2 = get_face_feats(face_feats_net, aligned_face_chips_ij[idxs_w_face_feats_from_search])
                    face_feats_target_2 = face_feats_model.semantic_search(face_feats_2, face_indicators_ij[idxs_w_face_feats_from_search])
                    loss_face_ij[idxs_w_face_feats_from_search] = (1 - (face_feats_2*face_feats_target_2).sum(dim=-1)).to(loss_face_ij.dtype)

                dynamic_weights = gen_dynamic_weights(face_indicators_ij, targets_ij, preds_gender_ori_ij, probs_gender_ori_ij, factor=args.factor1)
                loss_ij = loss_fair_ij + args.weight_loss_img * dynamic_weights * (loss_CLIP_ij + loss_DINO_ij) + args.weight_loss_face * loss_face_ij
                accelerator.backward(loss_ij.mean())

                with torch.no_grad():
                    loss_fair_i[idxs_ij] = loss_fair_ij.to(loss_fair_i.dtype)
                    loss_face_i[idxs_ij] = loss_face_ij.to(loss_face_i.dtype)
                    loss_CLIP_i[idxs_ij] = loss_CLIP_ij.to(loss_CLIP_i.dtype)
                    loss_DINO_i[idxs_ij] = loss_DINO_ij.to(loss_DINO_i.dtype)
                    loss_i[idxs_ij] = loss_ij.to(loss_i.dtype)
                    
            # for logging purpose, gather all losses to main_process
            accelerator.wait_for_everyone()
            loss_fair_all = customized_all_gather(loss_fair_i, accelerator)
            loss_face_all = customized_all_gather(loss_face_i, accelerator)
            loss_CLIP_all = customized_all_gather(loss_CLIP_i, accelerator)
            loss_DINO_all = customized_all_gather(loss_DINO_i, accelerator)
            loss_all = customized_all_gather(loss_i, accelerator)

            loss_all = loss_all[loss_fair_all!=-1]
            loss_fair_all = loss_fair_all[loss_fair_all!=-1]
            loss_face_all = loss_face_all[loss_face_all!=-1]

            if accelerator.is_main_process:
                logs_i["loss_fair"].append(loss_fair_all)
                logs_i["loss_face"].append(loss_face_all)
                logs_i["loss_CLIP"].append(loss_CLIP_all)
                logs_i["loss_DINO"].append(loss_DINO_all)
                logs_i["loss"].append(loss_all)
            
            # process logs
            if accelerator.is_main_process:
                for key in ["loss_fair", "loss_face", "loss_CLIP", "loss_DINO", "loss"]:
                    if logs_i[key] == []:
                        logs_i.pop(key)
                    else:
                        logs_i[key] = torch.cat(logs_i[key])
                for key in ["gender_gap", "gender_gap_abs", "gender_pred_between_0.2_0.8"]:
                    if logs_i[key] == []:
                        logs_i.pop(key)

            ##########################################################################
            # log process for training
            if accelerator.is_main_process:
                for key, values in logs_i.items():
                    if isinstance(values, list):
                        wandb_tracker.log({f"train_{key}": np.mean(values)}, step=global_step)
                    else:
                        wandb_tracker.log({f"train_{key}": values.mean().item()}, step=global_step)

                for key, values in log_imgs_i.items():
                    wandb_tracker.log({f"train_{key}":wandb.Image(
                            data_or_path=values[0],
                            caption=prompt_i,
                        )
                        },
                        step=global_step
                        )

            # model_sanity_print(prefix_embedding.module.token_embedding, "check No.1, prefix_embedding: after accelerator.backward()")
            model_sanity_print(prefix_embedding.token_embedding, "check No.1, prefix_embedding: after accelerator.backward()")

            # note that up till now grads are not synced
            # we mannually sync grads
            # accelerator.wait_for_everyone()
            grad_is_finite = True
            with torch.no_grad():
                # for p in prefix_embedding.module.token_embedding.parameters():
                for p in prefix_embedding.token_embedding.parameters():
                    if not torch.isfinite(p.grad).all():
                        grad_is_finite = False
                    torch.distributed.all_reduce(p.grad, torch.distributed.ReduceOp.SUM)
                    # N_backward is identical across processes
                    p.grad = p.grad / accelerator.num_processes / N_backward
                
            # model_sanity_print(prefix_embedding.module.token_embedding, "check No.2, prefix_embedding: after gradients allreduce & average")
            model_sanity_print(prefix_embedding.token_embedding, "check No.2, prefix_embedding: after gradients allreduce & average")

            if grad_is_finite:
                optimizer.step()
            else:
                accelerator.print(f"grads are not finite, skipped!")
            
            lr_scheduler.step()
            
            if grad_is_finite:
                # prefix_embedding_ema.step(  prefix_embedding.module.token_embedding.parameters() )
                prefix_embedding_ema.step(  prefix_embedding.token_embedding.parameters() )

            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                with torch.no_grad():
                    # param_norm = np.mean([p.norm().item() for p in prefix_embedding.module.token_embedding.parameters()])
                    param_norm = np.mean([p.norm().item() for p in prefix_embedding.token_embedding.parameters()])
                    param_ema_norm = np.mean([p.norm().item() for p in prefix_embedding_ema.shadow_params])
                    wandb_tracker.log({f"train_prefix_embedding_norm": param_norm}, step=global_step)
                    wandb_tracker.log({f"train_prefix_embedding_ema_norm": param_ema_norm}, step=global_step)

            if global_step % args.evaluate_every_n_iter == 0:
                evaluation_step(global_step)

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        name = "checkpoint_tmp"
                        clean_checkpoint(args.ckpts_save_dir, name, args.checkpoints_total_limit)

                    save_path = os.path.join(args.ckpts_save_dir, f"checkpoint_tmp-{global_step}")
                    accelerator.save_state(save_path)
                
                    logger.info(f"Accelerator checkpoint saved to {save_path}")

                if global_step % args.checkpointing_steps_long == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`

                    save_path = os.path.join(args.ckpts_save_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                
                    logger.info(f"Accelerator checkpoint saved to {save_path}")

            torch.cuda.empty_cache()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)





# accelerate launch --config_file configs/accelerate_config.yaml 1-main-debias.py --config configs/debias-token.yaml