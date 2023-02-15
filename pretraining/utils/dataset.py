# encoding: utf-8
import numpy as np
import glob
import os
from torch.utils.data import Dataset
from .cvtransforms import *
import torch
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import random
import torchvision

jpeg = TurboJPEG()



class LRWDataset_MyAug(Dataset):
    def __init__(self, phases, args):

        self.labels = os.listdir(args.dataset)       
        self.color_jitter = torchvision.transforms.ColorJitter(0.3,0.3,0.3) 

        data_list = [[] for _ in range(500)]
        self.phases = phases      
        self.args = args

        for (i, label) in enumerate(self.labels):
            for phase in phases:
                data_list[i] += glob.glob(os.path.join(args.dataset, label, phase, '*.pkl'))                
            random.shuffle(data_list[i])
        
        self.paired_data_list = []
        while True:
            r = list(range(500))
            random.shuffle(r)
            cnt = 0
            temp_file_list = []
            for i in r:
                if len(data_list[i])>=2:
                    temp_file_list.append(data_list[i].pop())
                    temp_file_list.append(data_list[i].pop())
                    cnt+=1
                else:
                    break
            if cnt < 500: # not enough data for 500 classes
                break
            else:
                self.paired_data_list += temp_file_list

    def __getitem__(self, idx):
        tensor = torch.load(self.paired_data_list[idx])                    
        
        inputs = tensor.get('video')
        inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
        inputs = np.stack(inputs, 0) / 255.0
        batch_img = inputs[:,:,:,0] # 29, h, w
        
        remaining_list = range(29)
        if('train' in self.phases):
            batch_img = RandomDistort(batch_img, self.args.max_magnitude)
            batch_img, remaining_list = RandomFrameDrop(batch_img, tensor.get('duration'))
            batch_img = RandomCrop(batch_img, shaking_prob=self.args.shaking_prob)
            batch_img = HorizontalFlip(batch_img) # prob 0.5
            batch_img = torch.FloatTensor(batch_img[:,np.newaxis,...]) # 29, 1 (C), h, w
            batch_img = self.color_jitter(batch_img)
        else:
            batch_img = CenterCrop(batch_img, (88, 88))
            batch_img = torch.FloatTensor(batch_img[:,np.newaxis,...])
        
        result = {} 
        result['video'] = batch_img
        result['label'] = tensor.get('label')
        result['duration'] = 1.0 * tensor.get('duration')[remaining_list]

        return result

    def __len__(self):
        return len(self.paired_data_list)

    def load_duration(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if(line.find('Duration') != -1):
                    duration = float(line.split(' ')[1])
        
        tensor = torch.zeros(29)
        mid = 29 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor            

