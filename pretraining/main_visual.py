
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os

import numpy as np
import time
from model import *
import torch.optim as optim 

from torch.cuda.amp import autocast, GradScaler
from model import cross_entropy
from utils import LRWDataset_MyAug as Dataset
import shutil


torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser.add_argument('--gpus', type=str, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--temperture', type=float, required=True)
parser.add_argument('--n_dimention', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--max_epoch', type=int, required=True)
parser.add_argument('--shaking_prob', type=float, required=True)
parser.add_argument('--max_magnitude', type=float, required=True)
parser.add_argument('--test', type=str2bool, required=True)

# load opts
parser.add_argument('--weights', type=str, required=False, default=None)

# save prefix
parser.add_argument('--save_prefix', type=str, required=True)

# dataset
parser.add_argument('--dataset', type=str, required=True)



args = parser.parse_args()
odd_indexes = [i for i in range(args.batch_size) if not i&1]
even_indexes = [i for i in range(args.batch_size) if i&1]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus




def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}                
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
    
    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
    print('miss matched params:',missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    for name, param in model.named_parameters():
        param.requires_grad = True
        # only fine tune the projection head if you dont have enough gpu memory
        if name in pretrained_dict.keys() and "prejection_head" not in name:
            param.requires_grad = False
    return model
    




video_model = VideoModel(args).cuda()
        
if(args.weights is not None):
    print('load weights')
    weight = torch.load(args.weights, map_location=torch.device('cpu'))    
    load_missing(video_model, weight.get('video_model'))
    
if len(args.gpus) > 1:
    print("parallelling model...")
    video_model =  nn.DataParallel(video_model)



lr = args.batch_size / 32.0 / torch.cuda.device_count() * args.lr
optimizer = optim.Adam(video_model.parameters(), lr = lr, weight_decay=1e-4)     
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=40, threshold=0.001)



def dataset2dataloader(dataset, batch_size, num_workers, shuffle=False, mode = "train"):
    collate_fn = pad_3d_sequence if mode == "train" else None
    loader =  DataLoader(
        dataset,
        batch_size = batch_size, 
        num_workers = num_workers,   
        shuffle = shuffle,
        drop_last = False,
        pin_memory=True,
        collate_fn=collate_fn,
        )
    return loader

def add_msg(msg, k, v):
    if(msg != ''):
        msg = msg + ','
    msg = msg + k.format(v)
    return msg    

def pad_3d_sequence(batch):
    video_seq = [b['video'] for b in batch]
    duration_seq = [b['duration'] for b in batch]
    lable_seq = [b['label'] for b in batch]

    video_seq = [torch.Tensor(b) for b in video_seq]
    video_seq_padded = torch.nn.utils.rnn.pad_sequence(
        video_seq, batch_first=True)
    duration_seq = [torch.Tensor(b) for b in duration_seq]
    duration_seq_padded = torch.nn.utils.rnn.pad_sequence(
        duration_seq, batch_first=True)
    labels = torch.LongTensor(lable_seq).squeeze()
    return {'video': video_seq_padded,
            'duration': duration_seq_padded,
            'label': labels}
    
def embed_from_path(npy_path):
    rois = np.load(str(npy_path))
    total_len = len(rois)
    rois = rois[np.newaxis,:,np.newaxis]
    buffer = torch.FloatTensor(rois/255.0).cuda()
    border = torch.FloatTensor([[1.0 for i in range(total_len)]]).cuda()

    with torch.no_grad():
        embedding = F.normalize(video_model(buffer,border),p=2, dim=-1) 
    return embedding.detach().cpu().numpy()

def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += ['{:.6f}'.format(param_group['lr'])]
    return ','.join(lr)

def train():
    
    max_epoch = args.max_epoch    
    best_loss = np.nan

    tot_iter = 0
    scaler = GradScaler()     
    tic = time.time()  
    for epoch in range(max_epoch):
        for phase in ['train', 'val']:
            dataset = Dataset([phase], args)
            print(f'Start {phase} phase, dataset size:',len(dataset))
            loader = dataset2dataloader(dataset, args.batch_size, args.num_workers, mode=phase)
            for (i_iter, input) in enumerate(loader):
                         
                batch_length = len(input.get('duration'))

                videos = input.get('video').cuda(non_blocking=True)
                borders = input.get('duration').cuda(non_blocking=True).float()

                with autocast():    
                    embeddings = video_model(videos,borders)          
                group_a_embeddings =  F.normalize(embeddings[odd_indexes[:batch_length//2]] , dim=-1) 
                group_b_embeddings = F.normalize(embeddings[even_indexes[:batch_length//2]] , dim=-1) 
                logits = (group_a_embeddings @ group_b_embeddings.T) / args.temperture
                loss = cross_entropy(logits, torch.eye(batch_length//2).cuda(), reduction='none')
                loss = loss.mean() 
                if phase == 'train':
                    optimizer.zero_grad()   
                    scaler.scale(loss).backward()  
                    scaler.step(optimizer)
                    scaler.update()

                    toc = time.time()
                    msg = 'epoch={}, train_iter={}, eta={:.1f}s'.format(epoch, tot_iter, (toc-tic)*(len(loader)-i_iter))     
                    tic = time.time()                                          
                    msg += ', train loss={:.5f}'.format(loss)
                    msg = msg + str(', lr=' + str(showLR(optimizer)))                 
                    msg = msg + str(', best valid loss={:2f}'.format(best_loss))
                    print(msg)
                    tot_iter += 1 
                else:
                    valid_loss = loss
                    savename = os.path.join(args.save_prefix, 'last.pt')
                    if not os.path.exists(args.save_prefix):
                        os.makedirs(args.save_prefix)
                    print("saving model at " + savename)
                    if len(args.gpus) > 1:
                        torch.save(
                            {
                                'video_model': video_model.module.state_dict(),
                            }, savename)  
                    else:
                        torch.save(
                            {
                                'video_model': video_model.state_dict(),
                            }, savename)       
                    if valid_loss < best_loss or np.isnan(best_loss):
                        shutil.copy(savename, savename.replace('last.pt', 'best.pt'))
                        best_loss = valid_loss    
                        print('best loss updated to {:.5f}'.format(best_loss))
                        scheduler.step(valid_loss)    
        
if(__name__ == '__main__'):
    train()

