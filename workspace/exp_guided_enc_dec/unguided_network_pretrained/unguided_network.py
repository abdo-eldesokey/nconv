import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import math

from scipy.stats import poisson
from scipy import signal


class CNN(nn.Module):

    def __init__(self, pos_fn=None, num_channels=2):
        super().__init__() 
        
        self.pos_fn = pos_fn
        
        self.navg1 = self.navg_layer((5,5), 3, 1, num_channels,'p', True)
        self.navg2 = self.navg_layer((5,5), 3, num_channels, num_channels,'p', True)
        self.navg3 = self.navg_layer((5,5), 3, num_channels, num_channels,'p', True)
        self.navg4 = self.navg_layer((1,1), 3, num_channels, 1,'p', True)
                            
        self.navg34 = self.navg_layer((3,3), 3, 2*num_channels, num_channels,'p', True)
        self.navg23 = self.navg_layer((3,3), 3, 2*num_channels, num_channels,'p', True)
        self.navg12 = self.navg_layer((3,3), 3, 2*num_channels, num_channels,'p', True)
                
        self.bias1 = nn.Parameter(torch.zeros(num_channels)+0.01)
        self.bias2 = nn.Parameter(torch.zeros(num_channels)+0.01)
        self.bias3 = nn.Parameter(torch.zeros(num_channels)+0.01)
        self.bias4 = nn.Parameter(torch.zeros(1)+0.01)
        
        self.bias34 = nn.Parameter(torch.zeros(num_channels)+0.01)
        self.bias23 = nn.Parameter(torch.zeros(num_channels)+0.01)
        self.bias12 = nn.Parameter(torch.zeros(num_channels)+0.01)
        
            
    def forward(self, x0, c0):  


        x1, c1 = self.navg_forward(self.navg1, c0, x0, self.bias1)
        
        x1, c1 = self.navg_forward(self.navg2, c1, x1, self.bias2)
        
        x1, c1 = self.navg_forward(self.navg3, c1, x1, self.bias3)        
        
        ds = 2 
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)
        x1_ds = torch.zeros(c1_ds.size()).cuda()
        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i,j,:,:] = x1[i,j,:,:].view(-1)[idx[i,j,:,:].view(-1)].view(idx.size()[2:])
        
        c1_ds /= 4
        

        
        x2_ds, c2_ds = self.navg_forward(self.navg2, c1_ds, x1_ds, self.bias2)
        
        x2_ds, c2_ds = self.navg_forward(self.navg3, c2_ds, x2_ds, self.bias3)
        
        
        
        ds = 2 
        c2_dss, idx = F.max_pool2d(c2_ds, ds, ds, return_indices=True)
        
        x2_dss = torch.zeros(c2_dss.size()).cuda()
        for i in range(x2_dss.size(0)):
            for j in range(x2_dss.size(1)):
                x2_dss[i,j,:,:] = x2_ds[i,j,:,:].view(-1)[idx[i,j,:,:].view(-1)].view(idx.size()[2:])
        c2_dss /= 4
        

        x3_ds, c3_ds = self.navg_forward(self.navg2, c2_dss, x2_dss, self.bias2)

        #x3_ds, c3_ds = self.navg_forward(self.navg3, c3_ds, x3_ds, self.bias3)
        
        
        ds = 2 
        c3_dss, idx = F.max_pool2d(c3_ds, ds, ds, return_indices=True)
        
        x3_dss = torch.zeros(c3_dss.size()).cuda()
        for i in range(x3_dss.size(0)):
            for j in range(x3_dss.size(1)):
                x3_dss[i,j,:,:] = x3_ds[i,j,:,:].view(-1)[idx[i,j,:,:].view(-1)].view(idx.size()[2:])
        c3_dss /= 4
        
        x4_ds, c4_ds = self.navg_forward(self.navg2, c3_dss, x3_dss, self.bias2)                

        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest') 
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest') 
        

        x34_ds, c34_ds = self.navg_forward(self.navg34, torch.cat((c3_ds,c4), 1),torch.cat((x3_ds,x4), 1), self.bias34)                
        
        
        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest') 
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest') 
        
        
        x23_ds, c23_ds = self.navg_forward(self.navg23, torch.cat((c2_ds,c34), 1),torch.cat((x2_ds,x34), 1), self.bias23)
        
        
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest') 
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest') 
        
            
        xout, cout = self.navg_forward(self.navg12, torch.cat((c23,c1), 1),torch.cat((x23,x1), 1), self.bias12)
        
        xout, cout = self.navg_forward(self.navg4, cout, xout, self.bias4)
        
        
        return xout, cout
        
    def navg_forward(self, navg, c, x, b, eps=1e-20, restore=False):
                            
        # Normalized Averaging 
        ca = navg(c) 
        xout = torch.div(navg(x*c), ca + eps)
        
        # Add bias
        sz = b.size(0)
        b = b.view(1,sz,1,1)
        b = b.expand_as(xout)
        xout = xout + b
        
        if restore:
            cm = (c == 0).float()
            xout = torch.mul(xout, cm) + torch.mul(1-cm, x)
            
        # Propagate confidence
        #cout = torch.ne(ca, 0).float()
        cout = ca
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)
        
        k = navg.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)
        
        cout = cout / s 
        
        cout = cout.view(sz)
        k = k.view(k_sz)
        
        return xout, cout
    
    
    def navg_layer(self, kernel_size, init_stdev=0.5, in_channels=1, out_channels=1, initalizer='x', pos=False, groups=1):
        
        navg = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, 
                         padding=(kernel_size[0]//2, kernel_size[1]//2), bias=False, groups=groups)
        
        weights = navg.weight            
        
        if initalizer == 'x': # Xavier            
            torch.nn.init.xavier_uniform(weights)
        elif initalizer == 'k':    
            torch.nn.init.kaiming_uniform(weights)
        elif initalizer == 'p':    
            mu=kernel_size[0]/2 
            dist = poisson(mu)
            x = np.arange(0, kernel_size[0])
            y = np.expand_dims(dist.pmf(x),1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.FloatTensor(w).cuda()
            w = torch.unsqueeze(w,0)
            w = torch.unsqueeze(w,1)
            w = w.repeat(out_channels, 1, 1, 1)
            w = w.repeat(1, in_channels, 1, 1)
            weights.data = w + torch.rand(w.shape).cuda()
         
        return navg
