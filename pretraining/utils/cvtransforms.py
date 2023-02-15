# coding: utf-8
import random
import numpy as np
import torch
import torch.nn.functional as F

import scipy.ndimage as ndi


def TensorRandomFlip(tensor):
    # (b, c, t, h, w)
    if(random.random() > 0.5):
        return torch.flip(tensor, dims=[4])        
    return tensor        

def TensorRandomCrop(tensor, size):
    h, w = tensor.size(-2), tensor.size(-1)
    tw, th = size
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return tensor[:,:,:,x1:x1+th, y1:y1+w]


def CenterCrop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    img = np.zeros((batch_img.shape[0], th, tw))
    x1 = int(round((w - tw))/2.)
    y1 = int(round((h - th))/2.)    
    img = batch_img[:, y1:y1+th, x1:x1+tw]
    return img

def RandomCrop(batch_img, shaking_prob = 0.2):
    w, h, l = batch_img.shape[2], batch_img.shape[1], batch_img.shape[0]
    th = random.randint(88,100)
    tw = th
    margin = (w - th) // 2 # margin = 20 if 128, 88, margin = 14 if 128, 100

    x1 = random.randint(int(margin*0.7), int(margin*1.3))
    y1 = random.randint(int(margin*0.7), int(margin*1.3))

    cropped_batch_img = batch_img[:,y1:y1+th,x1:x1+tw]
    # -- shake up to shaping_prob% frames with a prob of 50 %    
    if shaking_prob > 0 and random.random() < 0.3:
        for i in random.sample(range(l), random.randint(0,int(l * shaking_prob))):
            x1 = random.randint(int(margin*0.3), int(margin*1.7))
            y1 = random.randint(int(margin*0.3), int(margin*1.7))
            cropped_batch_img[i] = batch_img[i,y1:y1+th,x1:x1+tw]
    if th != 88:
        factor = 88/th
        cropped_batch_img = ndi.zoom(cropped_batch_img, (1, factor, factor), order=2)

    return cropped_batch_img

def HorizontalFlip(batch_img):
    if random.random() > 0.5:
        batch_img = np.ascontiguousarray(batch_img[:,:,::-1])
    return batch_img

def RandomFrameDrop(batch_img, duration):
    remaining_list = range(29)
    if random.random() > 0.5:
        drop_margin = int((29 - duration.sum() * 0.8 ) / 2) 
        drop_start = random.randint(0, drop_margin) 
        drop_end = random.randint(0, drop_margin)
        remaining_list = np.r_[drop_start:29-drop_end]
        batch_img = batch_img[remaining_list] 
    return batch_img, remaining_list
    

def get_of_fisheye(H, W, center, magnitude):  
    xx, yy = torch.linspace(-1, 1, W), torch.linspace(-1, 1, H)  
    gridy, gridx  = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack([gridx, gridy], dim=-1)  
    d = center - grid      
    d_sum = torch.sqrt((d**2).sum(axis=-1)) 
    grid += d * d_sum.unsqueeze(-1) * magnitude 
    return grid.unsqueeze(0)

def RandomDistort(batch_img, max_magnitude): 
    if random.random() > 0.5:
        w, h = batch_img.shape[2], batch_img.shape[1]
        center_x = (random.random() - 0.5) * 2
        center_y = random.random() * 0.25 - 1.5
        magnitude =  random.random() * max_magnitude
        fisheye_grid = get_of_fisheye(h, w, torch.tensor([center_x, center_y]), magnitude)
        fisheye_output = F.grid_sample(torch.FloatTensor(batch_img[None,...]), fisheye_grid,align_corners=False)
        return np.array(fisheye_output[0])
    else:
        return batch_img