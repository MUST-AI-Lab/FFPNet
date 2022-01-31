import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def _glcl(img1, img2, size_average=True):
    # T = 170
    T = 0.0026
    (_, channel, _, _) = img1.size()
    h_x = torch.tensor([[1./3,0,-1./3], [1./3,0,-1./3], [1./3,0,-1./3]]).unsqueeze(0).unsqueeze(0)
    h_x = h_x.expand(channel, 1, 3, 3).contiguous()
    h_y = h_x.transpose(2,3)
    if img1.is_cuda:
        h_x = h_x.cuda(img1.get_device())
        h_y = h_y.cuda(img1.get_device())
    # feat1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
    # feat2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
    feat1 = img1
    feat2 = img2

    feat1_x = F.conv2d(feat1, h_x, padding=1)
    feat1_y = F.conv2d(feat1, h_y, padding=1)
    grad1 = torch.sqrt(feat1_x.pow(2) + feat1_y.pow(2))

    feat2_x = F.conv2d(feat2, h_x, padding=1)
    feat2_y = F.conv2d(feat2, h_y, padding=1)
    grad2 = torch.sqrt(feat2_x.pow(2) + feat2_y.pow(2))

    gms = (2 * grad1 * grad2 + T) / (grad1.pow(2) + grad2.pow(2) + T)
    err = abs(img1 - img2)
    err = torch.log(torch.cosh(err) + (1-gms))
    if size_average:
        return err.mean()
    else:
        return err.sum()

class GLCL(torch.nn.Module):
    def __init__(self, size_average = True):
        super(GLCL, self).__init__()
        self.size_average = size_average

    def forward(self, img1, img2):
        return _glcl(img1, img2, self.size_average)

def glcl(img1, img2, size_average = True):
    return _glcl(img1, img2, size_average)

