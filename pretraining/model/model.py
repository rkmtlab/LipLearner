from .video_cnn import VideoCNN
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
        
class VideoModel(nn.Module):

    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()   
        self.args = args
        
        self.video_cnn = VideoCNN(se=True)        
        in_dim = 512 + 1
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)        

        self.prejection_head = nn.Linear(1024*2, args.n_dimention)     
        self.dropout = nn.Dropout(p=dropout)        

    def forward(self, v, border=None):
        self.gru.flatten_parameters()
        
        if(self.training):                            
            with autocast():
                f_v = self.video_cnn(v)  
                f_v = self.dropout(f_v)        
            f_v = f_v.float()
        else:                            
            f_v = self.video_cnn(v)  
            f_v = self.dropout(f_v)        
        
        border = border[:,:,None]
        h, _ = self.gru(torch.cat([f_v, border], -1))
        y_v = self.prejection_head(h).mean(1)

        return y_v


