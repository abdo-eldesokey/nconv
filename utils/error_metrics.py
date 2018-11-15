import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MAE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):     
        val_pixels = (target>0).float().cuda()
        err = torch.abs(target*val_pixels - outputs*val_pixels)
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss/cnt)

class RMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):     
        val_pixels = (target>0).float().cuda()
        err = (target*val_pixels - outputs*val_pixels)**2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(torch.sqrt(loss/cnt))
    
class MRE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):     
        val_pixels = (target>0).float().cuda()
        err = torch.abs(target*val_pixels - outputs*val_pixels)
        r = err / (target*val_pixels+1e-6)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        mre = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True) / cnt
        return torch.mean(mre)
    
class Deltas(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):     
        val_pixels = (target>0).float().cuda()
        rel = torch.max((target*val_pixels)/(outputs*val_pixels + 1e-3), (outputs*val_pixels) / (target*val_pixels))
        
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)

        def del_i(i):
            r = (rel < 1.01 ** i).float()
            delta = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True) / cnt
            return torch.mean(delta)
        
        return del_i(1), del_i(2), del_i(3)

    
class Huber(nn.Module):
    
    def __init__(self):
        super().__init__()
            
            
    def forward(self, outputs, target, delta=5):
        
        l1_loss = F.l1_loss(outputs, target, reduce=False)
        mse_loss = F.mse_loss(outputs, target, reduce=False)
        
        mask = (l1_loss < delta).float()
                
        loss = (0.5 * mse_loss) * mask + delta*(l1_loss - 0.5*delta) * (1-mask)
                
        return torch.mean(loss)

