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
from torch.nn import Conv2d
import numpy as np
import math
import importlib
import sys


class CNN(nn.Module):

    def __init__(self, pos_fn=None):
        super().__init__() 
        
        # Import the unguided network
        sys.path.append('workspace/exp_guided_enc_dec/unguided_network_pretrained')
        f = importlib.import_module('unguided_network')
        self.d_net = f.CNN()
        checkpoint_dict = torch.load('workspace/exp_guided_enc_dec/unguided_network_pretrained/CNN_ep0005.pth.tar')
        self.d_net.load_state_dict(checkpoint_dict['net'])
        
        # Disable Training for the unguided module
        for p in self.d_net.parameters():            
            p.requires_grad=False
        
        # U-Net
        self.conv1 = Conv2d(5, 80, (3,3), 2, 1, bias=True)
        self.conv2 = Conv2d(80, 80, (3,3), 2,1, bias=True)
        self.conv3 = Conv2d(80, 80, (3,3), 2, 1, bias=True)
        self.conv4 = Conv2d(80, 80, (3,3), 2, 1, bias=True)
        self.conv5 = Conv2d(80, 80, (3,3), 2, 1, bias=True)
                
        self.conv6 = Conv2d(80+80, 64, (3,3), 1, 1, bias=True)
        self.conv7 = Conv2d(64+80, 64, (3,3), 1, 1, bias=True)
        self.conv8 = Conv2d(64+80, 32, (3,3), 1, 1, bias=True)
        self.conv9 = Conv2d(32+80, 32, (3,3), 1, 1, bias=True)
        self.conv10 = Conv2d(32+1, 1, (3,3), 1, 1, bias=True)
        

            
        # Init Weights
        cc = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, \
        self.conv6, self.conv7, self.conv8, self.conv9, self.conv10,]
        for m in cc:            
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

               
            
    def forward(self, x0_d, c0, x0_rgb ):  
        
        # Depth Network
        xout_d, cout_d = self.d_net(x0_d, c0)

        # U-Net
        x1 = F.relu(self.conv1(torch.cat((xout_d, x0_rgb,cout_d),1)))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        # Upsample 1 
        x5u = F.interpolate(x5, x4.size()[2:], mode='nearest')
        x6 = F.leaky_relu(self.conv6(torch.cat((x5u, x4),1)), 0.2)
        
        # Upsample 2
        x6u = F.interpolate(x6, x3.size()[2:], mode='nearest')
        x7 = F.leaky_relu(self.conv7(torch.cat((x6u, x3),1)), 0.2)
        
        # Upsample 3
        x7u = F.interpolate(x7, x2.size()[2:], mode='nearest')
        x8 = F.leaky_relu(self.conv8(torch.cat((x7u, x2),1)), 0.2)
        
        # Upsample 4
        x8u = F.interpolate(x8, x1.size()[2:], mode='nearest')
        x9 = F.leaky_relu(self.conv9(torch.cat((x8u, x1),1)), 0.2)
                
        # Upsample 5
        x9u = F.interpolate(x9, x0_d.size()[2:], mode='nearest')
        xout = F.leaky_relu(self.conv10(torch.cat((x9u, x0_d),1)), 0.2)
        
        return xout, cout_d
       
