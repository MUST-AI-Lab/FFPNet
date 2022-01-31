import torch.nn as nn
import torch
from dct3d import DCT3D
from torchvision import models
from utils import save_net,load_net


def conv_block(inp, oup, dilation):
    return nn.Sequential(
        nn.Conv2d(512, 128, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=1),
        nn.ReLU(inplace=True),
    )

class DCTBlock(nn.Module):
    def __init__(self, inp, oup, freqs):
        super(DCTBlock, self).__init__()
        
        self.dct = DCT3D(freqs)

        self.branch_1 = nn.Sequential(
            conv_block(inp, oup, 2)
        )
        self.branch_2 = nn.Sequential(
            conv_block(inp, oup, 2)
        )
        self.branch_3 = nn.Sequential(
            conv_block(inp, oup, 2)
        )
        self.branch_4 = nn.Sequential(
            conv_block(inp, oup, 2)
        )
        self.weights = nn.Softmax()

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):

        d = self.dct(x)
        d_1 = self.branch_1(d[0])
        d_2 = self.branch_2(d[1])
        d_3 = self.branch_3(d[2])
        d_4 = self.branch_4(d[3])
        d = torch.cat((d_1, d_2, d_3, d_4), 1)
        w = self.weights(d)
        x = x * w + x
        return x


def make_layers(cfg, in_channels = 3, batch_norm=False, dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
        

class FFPNet(nn.Module):
    def __init__(self, load_weights=False):
        super(FFPNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
        self.backend_feat1  = [512]
        self.backend_feat2 = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend1 = []
        inp = 512
        freqs = [1, 16, 32, 64]
        for oup in self.backend_feat1:
            self.backend1.append(DCTBlock(inp, oup, freqs))
            inp = oup
        self.backend1 = nn.Sequential(*self.backend1)
        self.backend2 = make_layers(self.backend_feat2, in_channels = 512, dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in xrange(len(self.frontend.state_dict().items())):
                self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend1(x)
        x = self.backend2(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
