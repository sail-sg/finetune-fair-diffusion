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
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
import pytz
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torch import nn
import pickle as pkl
from skimage import transform
import kornia
import glob
from insightface.app import FaceAnalysis
import torchvision
from torchvision import transforms
import face_recognition

from PIL import Image, ImageOps, ImageDraw, ImageFont

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.19.0.dev0")

# testing against diffusers == 0.19.3

# os.environ["gpu_ids"] = "1"
my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.






def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def plot_in_grid_gender_race(images, save_to, face_indicators=None, face_bboxs=None, preds_gender=None, pred_class_probs_gender=None, preds_race=None, pred_class_probs_race=None):
    """
    images: torch tensor in shape of [N,3,H,W], in range [-1,1]
    """
    images_w_face = images[face_indicators]
    images_wo_face = images[face_indicators.logical_not()]

    # first reorder everything from most to least male, from most to least female, and finally images without faces
    idxs_male_white = ((preds_gender==1) * (preds_race == 0)).nonzero(as_tuple=False).view([-1])
    probs_male_white = pred_class_probs_race[idxs_male_white]
    idxs_male_white = idxs_male_white[probs_male_white.argsort(descending=True)]

    idxs_male_black = ((preds_gender==1) * (preds_race == 1)).nonzero(as_tuple=False).view([-1])
    probs_male_black = pred_class_probs_race[idxs_male_black]
    idxs_male_black = idxs_male_black[probs_male_black.argsort(descending=True)]
    
    idxs_male_indian = ((preds_gender==1) * (preds_race == 2)).nonzero(as_tuple=False).view([-1])
    probs_male_indian = pred_class_probs_race[idxs_male_indian]
    idxs_male_indian = idxs_male_indian[probs_male_indian.argsort(descending=True)]
    
    idxs_male_asian = ((preds_gender==1) * (preds_race == 3)).nonzero(as_tuple=False).view([-1])
    probs_male_asian = pred_class_probs_race[idxs_male_asian]
    idxs_male_asian = idxs_male_asian[probs_male_asian.argsort(descending=True)]
    
    idxs_female_white = ((preds_gender==0) * (preds_race == 0)).nonzero(as_tuple=False).view([-1])
    probs_female_white = pred_class_probs_race[idxs_female_white]
    idxs_female_white = idxs_female_white[probs_female_white.argsort(descending=True)]

    idxs_female_black = ((preds_gender==0) * (preds_race == 1)).nonzero(as_tuple=False).view([-1])
    probs_female_black = pred_class_probs_race[idxs_female_black]
    idxs_female_black = idxs_female_black[probs_female_black.argsort(descending=True)]
    
    idxs_female_indian = ((preds_gender==0) * (preds_race == 2)).nonzero(as_tuple=False).view([-1])
    probs_female_indian = pred_class_probs_race[idxs_female_indian]
    idxs_female_indian = idxs_female_indian[probs_female_indian.argsort(descending=True)]
    
    idxs_female_asian = ((preds_gender==0) * (preds_race == 3)).nonzero(as_tuple=False).view([-1])
    probs_female_asian = pred_class_probs_race[idxs_female_asian]
    idxs_female_asian = idxs_female_asian[probs_female_asian.argsort(descending=True)]

    idxs_no_face = (preds_race == -1).nonzero(as_tuple=False).view([-1])

    images_to_plot = []
    idxs_reordered = torch.torch.cat([idxs_male_white, idxs_male_black, idxs_male_indian, idxs_male_asian, idxs_female_white, idxs_female_black, idxs_female_indian, idxs_female_asian, idxs_no_face])
    
    for idx in idxs_reordered:
        img = images[idx]
        face_indicator = face_indicators[idx]
        face_bbox = face_bboxs[idx]
        pred_gender = preds_gender[idx]
        pred_class_prob_gender = pred_class_probs_gender[idx]
        pred_race = preds_race[idx]
        pred_class_prob_race = pred_class_probs_race[idx]
        
        if pred_gender == 0:
            gender_border_color = "red"
        elif pred_gender == 1:
            gender_border_color = "blue"
        elif pred_gender == -1:
            gender_border_color = "white"
        if pred_race == 0:
            race_border_color = "limegreen"
        elif pred_race == 1:
            race_border_color = "Black"
        elif pred_race == 2:
            race_border_color = "brown"
        elif pred_race == 3:
            race_border_color = "orange"
        elif pred_race == -1:
            race_border_color = "white"

        img_pil = transforms.ToPILImage()(img*0.5+0.5)
        img_pil_draw = ImageDraw.Draw(img_pil)  
        img_pil_draw.rectangle(face_bbox.tolist(), fill =None, outline ="black", width=4)

        img_pil = ImageOps.expand(img_pil_draw._image, border=(50,0,0,0),fill=race_border_color)
        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_race.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_race.item())*512)], fill ="white", outline =None)
            
        img_pil = ImageOps.expand(img_pil_draw._image, border=(50,0,0,0),fill=gender_border_color)
        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_gender.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_gender.item())*512)], fill ="white", outline =None)
            
        fnt = ImageFont.truetype(font="./data/0-utils/arial-bold.ttf", size=100)
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


def plot_in_grid_gender_race_age(images, save_to, face_indicators=None, face_bboxs=None, preds_gender=None, pred_class_probs_gender=None, preds_race=None, pred_class_probs_race=None, preds_age=None, pred_class_probs_age=None):
    """
    images: torch tensor in shape of [N,3,H,W], in range [-1,1]
    """

    idxs_reordered = []
    for g in [1,0]:
        for r in [0,1,2,3]:
            for a in [0,1]:
                idxs_ = ((preds_gender==g) * (preds_race == r) * (preds_age == a)).nonzero(as_tuple=False).view([-1])
                probs_ = pred_class_probs_gender[idxs_]
                idxs_ = idxs_[probs_.argsort(descending=True)]
                idxs_reordered.append(idxs_)
                
    idxs_no_face = (preds_race == -1).nonzero(as_tuple=False).view([-1])
    idxs_reordered.append(idxs_no_face)    
    idxs_reordered = torch.cat(idxs_reordered) 

    images_to_plot = []
    for idx in idxs_reordered:
        img = images[idx]
        face_indicator = face_indicators[idx]
        face_bbox = face_bboxs[idx]
        pred_gender = preds_gender[idx]
        pred_class_prob_gender = pred_class_probs_gender[idx]
        pred_race = preds_race[idx]
        pred_class_prob_race = pred_class_probs_race[idx]
        pred_age = preds_age[idx]
        pred_class_prob_age = pred_class_probs_age[idx]
        
        if pred_gender == 0:
            gender_border_color = "red"
        elif pred_gender == 1:
            gender_border_color = "blue"
        elif pred_gender == -1:
            gender_border_color = "white"

        if pred_race == 0:
            race_border_color = "limegreen"
        elif pred_race == 1:
            race_border_color = "Black"
        elif pred_race == 2:
            race_border_color = "brown"
        elif pred_race == 3:
            race_border_color = "orange"
        elif pred_race == -1:
            race_border_color = "white"
        
        if pred_age == 0:
            age_border_color = "darkorange"
        elif pred_age == 1:
            age_border_color = "darkgreen"
        elif pred_age == -1:
            age_border_color = "white"

        img_pil = transforms.ToPILImage()(img*0.5+0.5)
        img_pil_draw = ImageDraw.Draw(img_pil)  
        img_pil_draw.rectangle(face_bbox.tolist(), fill =None, outline ="black", width=4)

        img_pil = ImageOps.expand(img_pil_draw._image, border=(50,0,0,0),fill=age_border_color)
        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_race.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_age.item())*512)], fill ="white", outline =None)
            
        img_pil = ImageOps.expand(img_pil_draw._image, border=(50,0,0,0),fill=race_border_color)
        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_race.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_race.item())*512)], fill ="white", outline =None)
            
        img_pil = ImageOps.expand(img_pil_draw._image, border=(50,0,0,0),fill=gender_border_color)
        img_pil_draw = ImageDraw.Draw(img_pil)
        if pred_class_prob_gender.item() < 1:
            img_pil_draw.rectangle([(0,0),(50,(1-pred_class_prob_gender.item())*512)], fill ="white", outline =None)
            
        fnt = ImageFont.truetype(font="./data/0-utils/arial-bold.ttf", size=100)
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

def image_pipeline(img, tgz_landmark):
    img = (img+1)/2.0 * 255 # map to [0,255]

    crop_size = (112,112)
    src_landmark = np.array(
    [[38.2946, 51.6963], # 左眼
    [73.5318, 51.5014], # 右眼
    [56.0252, 71.7366], # 鼻子
    [41.5493, 92.3655], # 左嘴角
    [70.7299, 92.2041]] # 右嘴角
    )

    tform = transform.SimilarityTransform()
    tform.estimate(tgz_landmark, src_landmark)

    M = torch.tensor(tform.params[0:2, :]).unsqueeze(dim=0).to(img.dtype).to(img.device)
    img_face = kornia.geometry.transform.warp_affine(img.unsqueeze(dim=0), M, crop_size, mode='bilinear', padding_mode='zeros', align_corners=False)
    img_face = img_face.squeeze()

    img_face = (img_face/255.0)*2-1 # map back to [-1,1]
    return img_face

def get_face(images, face_app, face_recognition, size_face=224, size_aligned_face=112, fill_value=-1):
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
    face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app = get_face_app(images, face_app, size_face, size_aligned_face, fill_value=fill_value)

    if face_indicators_app.logical_not().sum() > 0:
        face_indicators_FR, face_bboxs_FR, face_chips_FR, face_landmarks_FR, aligned_face_chips_FR = get_face_FR(images[face_indicators_app.logical_not()], face_recognition, size_face, size_aligned_face, fill_value=fill_value)

        face_bboxs_app[face_indicators_app.logical_not()] = face_bboxs_FR
        face_chips_app[face_indicators_app.logical_not()] = face_chips_FR
        face_landmarks_app[face_indicators_app.logical_not()] = face_landmarks_FR
        aligned_face_chips_app[face_indicators_app.logical_not()] = aligned_face_chips_FR

        face_indicators_app[face_indicators_app.logical_not()] = face_indicators_FR

    return face_indicators_app, face_bboxs_app, face_chips_app, face_landmarks_app, aligned_face_chips_app

def get_face_app(images, face_app, size_face, size_aligned_face, fill_value=-1):
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
            face_chips_app.append(torch.ones([1,3,size_face,size_face], dtype=images.dtype, device=images.device)*(fill_value))
            face_landmarks_app.append(torch.ones([1,5,2], dtype=images.dtype, device=images.device)*(fill_value))
            aligned_face_chips_app.append(torch.ones([1,3,size_aligned_face,size_aligned_face], dtype=images.dtype, device=images.device)*(fill_value))
        else:
            face_from_app = get_largest_face_app(faces_from_app, dim_max=image_np.shape[0], dim_min=0)
            bbox = expand_bbox(face_from_app["bbox"], expand_coef=0.5, target_ratio=1)
            face_chip = crop_face(images[idx], bbox, target_size=[size_face,size_face], fill_value=fill_value)
            
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


def get_face_FR(images, face_recognition, size_face, size_aligned_face, fill_value=-1):
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
            face_chips_FR.append(torch.ones([1,3,size_face,size_face], dtype=images.dtype, device=images.device)*(fill_value))
            face_landmarks_FR.append(torch.ones([1,5,2], dtype=images.dtype, device=images.device)*(fill_value))
            aligned_face_chips_FR.append(torch.ones([1,3,size_aligned_face,size_aligned_face], dtype=images.dtype, device=images.device)*(fill_value))
        else:
            face_from_FR = get_largest_face_FR(faces_from_FR, dim_max=image_np.shape[0], dim_min=0)
            bbox = face_from_FR
            bbox = np.array((bbox[-1],) + bbox[:-1]) # need to convert bbox from face_recognition to the right order
            bbox = expand_bbox(bbox, expand_coef=1.1, target_ratio=1) # need to use a larger expand_coef for FR
            face_chip = crop_face(images[idx], bbox, target_size=[size_face,size_face], fill_value=fill_value)
            
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


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script to finetune Stable Diffusion for debiasing purposes.")

    parser.add_argument(
        "--gpu_id", 
        type=int,
        default=0,
    )
    parser.add_argument(
        "--gender_classifier_weight",
        type=str,
        default="./data/5-trained-test-classifiers/CelebA-MobileNetLarge-Gender-09191318/epoch=19-step=25320_MobileNetLarge.pt",
        help="path to separately trained gender classifier for testing",
    )
    parser.add_argument(
        "--race_classifier_weight",
        type=str,
        default="./data/5-trained-test-classifiers/fairface-MobileNetLarge-Race4-09191318/epoch=19-step=6760_MobileNetLarge.pt",
        help="path to separately trained race classifier for testing",
    )
    parser.add_argument(
        "--age_classifier_weight",
        type=str,
        default="./data/5-trained-test-classifiers/fairface-MobileNetLarge-Age2-09191319/epoch=19-step=6760_MobileNetLarge.pt",
        help="path to separately trained age classifier for testing",
    )
    parser.add_argument(
        "--generated_imgs_dir",
        type=str,
        default="./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_occupation",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        # default=None,
        default="./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_occupation_results",
        # required=True
    )
    parser.add_argument(
        '--batch_size', 
        help="batch size for image generation", 
        type=int, 
        default=10
        )
    parser.add_argument(
        '--size_face', 
        help="size for extracted face", 
        type=int, 
        default=224
        )
    parser.add_argument(
        '--size_aligned_face', 
        help="size for aligned extracted face", 
        type=int, 
        default=112
        )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):

    args.device = f"cuda:{args.gpu_id}"

    face_app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=['detection'], 
        providers=['CUDAExecutionProvider'], 
        provider_options=[{'device_id': args.gpu_id}]
        )
    face_app.prepare(ctx_id=0, det_size=(640, 640))


    gender_classifier = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT, width_mult=1.0, reduced_tail=False, dilated=False)
    gender_classifier._modules['classifier'][3] = nn.Linear(1280, 2, bias=True)
    gender_classifier.load_state_dict(torch.load(args.gender_classifier_weight))
    gender_classifier.to(args.device)
    gender_classifier.requires_grad_(False)
    gender_classifier.eval()

    race_classifier = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT, width_mult=1.0, reduced_tail=False, dilated=False)
    race_classifier._modules['classifier'][3] = nn.Linear(1280, 4, bias=True)
    race_classifier.load_state_dict(torch.load(args.race_classifier_weight))
    race_classifier.to(args.device)
    race_classifier.requires_grad_(False)
    race_classifier.eval()

    age_classifier = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT, width_mult=1.0, reduced_tail=False, dilated=False)
    age_classifier._modules['classifier'][3] = nn.Linear(1280, 2, bias=True)
    age_classifier.load_state_dict(torch.load(args.age_classifier_weight))
    age_classifier.to(args.device)
    age_classifier.requires_grad_(False)
    age_classifier.eval()

    prompt_folders = sorted(
        glob.glob(os.path.join(args.generated_imgs_dir, "prompt_*")), 
        key=lambda x: int(x.split("_")[-1])
        )
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    face_indicators_all = {}
    face_bboxs_all = {}
    face_chips_all = {}
    aligned_face_chips_all = {}
    gender_logits_all = {}
    race_logits_all = {}
    age_logits_all = {}
    for prompt_folder in prompt_folders:
        prompt_idx = int(prompt_folder.split("_")[-1])

        imgs_p = []
        face_indicators_p = []
        face_bboxs_p = []
        face_chips_p = []
        aligned_face_chips_p = []
        gender_logits_p = []
        race_logits_p = []
        age_logits_p = []

        img_paths = sorted(
            glob.glob(os.path.join(prompt_folder, "img_*.jpg")),
            key=lambda x: int(x.split("_")[-1].split(".")[0])
            )

        for img_path in tqdm(img_paths):
            img = torchvision.io.read_image(img_path)
            img = img.float().unsqueeze(dim=0)/255*2-1
            
            face_indicator, face_bbox, face_chip, face_landmark, aligned_face_chip = get_face(img, face_app, face_recognition, args.size_face, args.size_aligned_face)
            
            face_chip = face_chip.to(args.device)
            
            gender_logit = gender_classifier(face_chip)
            race_logit = race_classifier(face_chip)
            age_logit = age_classifier(face_chip)
            
            imgs_p.append(img)
            face_indicators_p.append( face_indicator )
            face_bboxs_p.append( face_bbox )
            face_chips_p.append( face_chip )
            aligned_face_chips_p.append( aligned_face_chip )
            gender_logits_p.append( gender_logit )
            race_logits_p.append( race_logit )
            age_logits_p.append( age_logit )

        imgs_p = torch.cat( imgs_p )
        face_indicators_p = torch.cat( face_indicators_p )
        face_bboxs_p = torch.cat( face_bboxs_p )
        face_chips_p = torch.cat( face_chips_p )
        aligned_face_chips_p = torch.cat( aligned_face_chips_p )
        gender_logits_p = torch.cat( gender_logits_p )
        race_logits_p = torch.cat( race_logits_p )
        age_logits_p = torch.cat( age_logits_p )

        gender_probs_p = torch.softmax(gender_logits_p, dim=-1)
        race_probs_p = torch.softmax(race_logits_p, dim=-1)
        age_probs_p = torch.softmax(age_logits_p, dim=-1)
        gender_preds_p = gender_probs_p.max(dim=-1).indices
        race_preds_p = race_probs_p.max(dim=-1).indices
        age_preds_p = age_probs_p.max(dim=-1).indices

        save_to = os.path.join(args.save_dir, f"prompt_{prompt_idx}.jpg")
        plot_in_grid_gender_race(
            imgs_p, 
            save_to, 
            face_indicators=face_indicators_p, 
            face_bboxs=face_bboxs_p, 
            preds_gender=gender_preds_p, 
            pred_class_probs_gender=gender_probs_p.max(dim=-1).values,
            preds_race=race_preds_p, 
            pred_class_probs_race=race_probs_p.max(dim=-1).values,
        )
        # plot_in_grid_gender_race_age(
        #     imgs_p, 
        #     save_to, 
        #     face_indicators=face_indicators_p, 
        #     face_bboxs=face_bboxs_p, 
        #     preds_gender=gender_preds_p, 
        #     pred_class_probs_gender=gender_probs_p.max(dim=-1).values,
        #     preds_race=race_preds_p, 
        #     pred_class_probs_race=race_probs_p.max(dim=-1).values,
        #     preds_age=age_preds_p, 
        #     pred_class_probs_age=age_probs_p.max(dim=-1).values,
        # )

        face_indicators_all[prompt_idx] = face_indicators_p.cpu()
        face_bboxs_all[prompt_idx] = face_bboxs_p.cpu()
        face_chips_all[prompt_idx] = face_chips_p.cpu()
        aligned_face_chips_all[prompt_idx] = aligned_face_chips_p.cpu()
        gender_logits_all[prompt_idx] = gender_logits_p.cpu()
        race_logits_all[prompt_idx] = race_logits_p.cpu()
        age_logits_all[prompt_idx] = age_logits_p.cpu()

    # import pdb; pdb.set_trace()

    results = [face_indicators_all, face_bboxs_all, gender_logits_all, race_logits_all, age_logits_all]
    save_to = os.path.join(args.save_dir, "test_results.pkl")
    with open(save_to, "wb") as f:
        pkl.dump(results, f)
    

    





if __name__ == "__main__":
    args = parse_args()
    main(args)



"""
python eval-generated-images.py \
    --generated_imgs_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_occupation \
    --save_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_occupation_results

python eval-generated-images.py \
    --generated_imgs_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_occupation_w_style_and_context \
    --save_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_occupation_w_style_and_context_results

python eval-generated-images.py \
    --generated_imgs_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_personal_descriptor \
    --save_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_personal_descriptor_results

python eval-generated-images.py \
    --generated_imgs_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_sports \
    --save_dir ./exp-3-debias-gender-race/outputs/from-paper_finetune-text-encoder_09190230/checkpoint-12200-generated-images/test_prompts_sports_results
"""





"""
python eval-generated-images.py \
    --generated_imgs_dir ./original-SD-generated-images/test_prompts_occupation \
    --save_dir ./original-SD-generated-images/test_prompts_occupation_results

python eval-generated-images.py \
    --generated_imgs_dir ./original-SD-generated-images/test_prompts_occupation_w_style_and_context \
    --save_dir ./original-SD-generated-images/test_prompts_occupation_w_style_and_context_results

python eval-generated-images.py \
    --generated_imgs_dir ./original-SD-generated-images/test_prompts_personal_descriptor \
    --save_dir ./original-SD-generated-images/test_prompts_personal_descriptor_results

python eval-generated-images.py \
    --generated_imgs_dir ./original-SD-generated-images/test_prompts_sports \
    --save_dir ./original-SD-generated-images/test_prompts_sports_results
"""