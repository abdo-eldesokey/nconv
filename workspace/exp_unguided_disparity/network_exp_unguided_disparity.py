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
import numpy as np

from modules.nconv import NConv2d


class CNN(nn.Module):

    def __init__(self, pos_fn=None, num_channels=2):
        super().__init__() 
        
        self.pos_fn = pos_fn

        self.nconv1 = NConv2d(1, num_channels, (5,5), pos_fn, 'p', padding=2)
        self.nconv2 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=2)
        self.nconv3 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding=2)
        
        self.nconv4 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=1)
        self.nconv5 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=1)
        self.nconv6 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding=1)
        
        self.nconv7 = NConv2d(num_channels, 1, (1,1), pos_fn, 'k')
        
        
    def forward(self, x0, c0):  
   
        x1, c1 = self.nconv1(x0, c0)
        x1, c1 = self.nconv2(x1, c1)
        x1, c1 = self.nconv3(x1, c1)           
                
        # Downsample 1
        ds = 2 
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)
        x1_ds = torch.zeros(c1_ds.size()).cuda()
        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i,j,:,:] = x1[i,j,:,:].view(-1)[idx[i,j,:,:].view(-1)].view(idx.size()[2:])
        c1_ds /= 4
        
        x2_ds, c2_ds = self.nconv2(x1_ds, c1_ds)        
        x2_ds, c2_ds = self.nconv3(x2_ds, c2_ds)
        
        
        # Downsample 2
        ds = 2 
        c2_dss, idx = F.max_pool2d(c2_ds, ds, ds, return_indices=True)
        
        x2_dss = torch.zeros(c2_dss.size()).cuda()
        for i in range(x2_dss.size(0)):
            for j in range(x2_dss.size(1)):
                x2_dss[i,j,:,:] = x2_ds[i,j,:,:].view(-1)[idx[i,j,:,:].view(-1)].view(idx.size()[2:])
        c2_dss /= 4        

        x3_ds, c3_ds = self.nconv2(x2_dss, c2_dss)
        
        
        # Downsample 3
        ds = 2 
        c3_dss, idx = F.max_pool2d(c3_ds, ds, ds, return_indices=True)
        
        x3_dss = torch.zeros(c3_dss.size()).cuda()
        for i in range(x3_dss.size(0)):
            for j in range(x3_dss.size(1)):
                x3_dss[i,j,:,:] = x3_ds[i,j,:,:].view(-1)[idx[i,j,:,:].view(-1)].view(idx.size()[2:])
        c3_dss /= 4        
        x4_ds, c4_ds = self.nconv2(x3_dss, c3_dss)                


        # Upsample 1
        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest') 
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest')       
        x34_ds, c34_ds = self.nconv4(torch.cat((x3_ds,x4), 1),  torch.cat((c3_ds,c4), 1))                
        
        # Upsample 2
        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest') 
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest') 
        x23_ds, c23_ds = self.nconv5(torch.cat((x2_ds,x34), 1), torch.cat((c2_ds,c34), 1))
        
        
        # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest') 
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest') 
        xout, cout = self.nconv6(torch.cat((x23,x1), 1), torch.cat((c23,c1), 1))
        
        
        xout, cout = self.nconv7(xout, cout)
                
        return xout, cout
        
