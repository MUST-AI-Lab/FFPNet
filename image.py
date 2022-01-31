import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import scipy.io

def load_data(img_path,train = True):
    img = Image.open(img_path).convert('RGB')
    # gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    # gt_file = h5py.File(gt_path, 'r')
    # target = np.asarray(gt_file['density'])

    # gt_path = img_path.replace('.jpg','.mat').replace('images','labels_8_pt')
    gt_path = img_path.replace('.jpg','.mat').replace('images','labels')
    gt_file = scipy.io.loadmat(gt_path)
    target = np.asarray(gt_file['Heat_Map'])

    if False:
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1:
            
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
        
        
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    
    
    # target = cv2.resize(target,(target.shape[1]/8,target.shape[0]/8),interpolation = cv2.INTER_CUBIC)*64
    
    
    return img,target
