########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class InputOutputLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
                      
    def forward(self, outputs, target, cout, epoch_num, inputs, *args):           
        val_pixels = torch.ne(target, 0).float().cuda()
        err = F.smooth_l1_loss(outputs*val_pixels, target*val_pixels, reduction='none')
        
        val_pixels = torch.ne(inputs, 0).float().cuda()
        inp_loss = F.smooth_l1_loss(outputs*val_pixels, inputs*val_pixels, reduction='none')
        
        loss = err + 0.1 * inp_loss
        
        return torch.mean(loss)



class ConfLossDecay(nn.Module):
    
    def __init__(self):
        super().__init__()
                      
    def forward(self, outputs, target, cout, epoch_num, *args):           
        val_pixels = torch.ne(target, 0).float().cuda()
        err = F.smooth_l1_loss(outputs*val_pixels, target*val_pixels, reduction='none')
        cert = cout*val_pixels - err*cout*val_pixels
        loss = err - (1/epoch_num) * cert
        return torch.mean(loss)

class ConfLossDecayMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, cout, epoch_num, *args):           
        val_pixels = torch.ne(target, 0).float().cuda()
        err = F.mse_loss(outputs*val_pixels, target*val_pixels, reduction='none')
        cert = cout*val_pixels - err*cout*val_pixels
        loss = err - (1/epoch_num) * cert
        return torch.mean(loss)
    

class ConfLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
                      
    def forward(self, outputs, target, cout, *args):    
        val_pixels = torch.ne(target, 0).float().cuda()
        err = F.smooth_l1_loss(outputs*val_pixels, target*val_pixels, reduction='none')
        loss = err-cout*val_pixels+err*cout*val_pixels
        return torch.mean(loss)


class SmoothL1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = F.smooth_l1_loss(outputs*val_pixels, target*val_pixels, reduction='none')
        return torch.mean(loss)

    
    
class RMSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):       
        val_pixels = (target>0).float().cuda()
        err = (target*val_pixels - outputs*val_pixels)**2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(torch.sqrt(loss/cnt))
 
    
    
class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target*val_pixels - outputs*val_pixels
        return torch.mean(loss**2)
    

        
