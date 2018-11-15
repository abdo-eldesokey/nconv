import torch
import os
import cv2 
import numpy as np

# This function takes a 4D tensor in the form NxCxWxH and save it to images according to idxs
def saveTensorToImages(t, idxs, save_to_path):
    
    if os.path.exists(save_to_path)==False:
        os.mkdir(save_to_path)
    
    for i in range(t.size(0)):
        im = t[i,:,:,:].detach().data.cpu().numpy() 
        im = np.transpose(im, (1,2,0)).astype(np.uint16)
        cv2.imwrite(os.path.join(save_to_path, str(idxs[i].data.cpu().numpy()).zfill(10)+'.png'), im, [cv2.IMWRITE_PNG_COMPRESSION, 4] )
        
        
