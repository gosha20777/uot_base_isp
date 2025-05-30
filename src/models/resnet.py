import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """
    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
    

class ResNet_G(nn.Module):
    "Generator ResNet architecture from https://github.com/harryliew/WGAN-QC"
    def __init__(self, z_dim, size, nc=3, nfilter=64, nfilter_max=512, bn=True, res_ratio=0.1, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.bn = bn
        self.z_dim = z_dim
        self.nc = nc

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**(nlayers+1))

        self.fc = nn.Linear(z_dim, self.nf0*s0*s0)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0*s0*s0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2**(i+1), nf_max)
            nf1 = min(nf * 2**i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2)
            ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio)
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, nc, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)
        out = self.fc(z)
        if self.bn:
            out = self.bn1d(out)
        out = self.relu(out)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(out)
        out = torch.tanh(out)

        return out


class ResNet_D(nn.Module):
    "Discriminator ResNet architecture from https://github.com/harryliew/WGAN-QC"
    def __init__(self, size=64, nc=3, nfilter=64, nfilter_max=512, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.nc = nc

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=True, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=True, res_ratio=res_ratio)
        ]

        for i in range(1, nlayers+1):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf0, bn=True, res_ratio=res_ratio),
                ResNetBlock(nf0, nf1, bn=True, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(nc, 1*nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.relu((self.conv_img(x)))
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(out)

        return out


class ResNet_D_Progressive(nn.Module):
    def __init__(self, size=64, nc=3, nfilter=64, nfilter_max=512, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.nc = nc
        
        ##which nfilter?

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)
        ]

        for i in range(1, nlayers+1):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.convs_img = nn.ModuleList([nn.Conv2d(nc, min(2**(sz+4), self.nf0), 3, padding=1) for sz in range(2, int(np.log2(size))+1)])
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.resnet = nn.ModuleList(blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)

        # self.alpha = 0.5

    def forward(self, x, alpha):
        batch_size = x.size(0)

        n_inp_ch = x.size(1)
        resolution = int(np.log2(x.size(2))) 

        out = self.relu(self.convs_img[-(resolution - 1)](x)) # add max if work with resolution and num_filters > max_filters
        if resolution == 2: # for 4x4
            new_out = self.resnet[-1](self.resnet[-2](out))
            
        else: # 8x8 and up
            ind = (resolution - 2) * 3 + 2
            old_out = F.avg_pool2d(x, kernel_size=2, stride=2)
            old_out = self.relu(self.convs_img[-(resolution - 2)](old_out))

            for layer in range(-ind, -ind + 3):
                out = self.resnet[layer](out)

            # add
            new_out = (1 - alpha) * old_out + alpha * out

            for layer in range(-ind + 3, 0):
                new_out = self.resnet[layer](new_out)

        new_out = new_out.view(batch_size, self.nf0*self.s0*self.s0)
        new_out = self.fc(new_out)

        return new_out


class ResNetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = PixelNormLayer()
            # self.bn2d_0 = nn.BatchNorm2d(self.fhidden)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = PixelNormLayer()
            # self.bn2d_1 = nn.BatchNorm2d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = PixelNormLayer()
                # self.bn2d_s = nn.BatchNorm2d(self.fout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio*dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s


class ResDownBlock(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super().__init__()
        # Attributes

        self.fin = fin
        self.fout = fout
        self.fhidden = min(fin, fout)
        
        
        self.conv_res = nn.Conv2d(self.fin, self.fout, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.downsample = nn.Sequential(
#             Blur(),
            nn.Conv2d(self.fout, self.fout, 3, padding=1, stride=2)
        ) if downsample else None
        
    
    def forward(self, x):
        res = self.conv_res(x)

        x = self.net(x)
        if self.downsample:
            x = self.downsample(x)

        x = (x + res) * (1 / math.sqrt(2))
        return x


class ResDown_D(nn.Module):
    def __init__(self, size=64, nc=3, nfilter=64, nfilter_max=512):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.nc = nc
        
        ##which nfilter?

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResDownBlock(nf0, nf1)
        ]

        for i in range(1, nlayers+1):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                ResDownBlock(nf0, nf1, downsample=(True if i != nlayers else False))
            ]

        self.convs_img = nn.ModuleList([nn.Conv2d(nc, min(2**(sz+4), self.nf0), 3, padding=1) for sz in range(2, int(np.log2(size))+1)])
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.resnet = nn.ModuleList(blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)

        # self.alpha = 0.5

    def forward(self, x, alpha):
        batch_size = x.size(0)

        n_inp_ch = x.size(1)
        resolution = int(np.log2(x.size(2))) 

        out = self.relu(self.convs_img[-(resolution - 1)](x)) # add max if work with resolution and num_filters > max_filters
        if resolution == 2: # for 4x4
            new_out = self.resnet[-1](out)
            
        else: # 8x8 and up
            ind = (resolution - 1)
            old_out = F.avg_pool2d(x, kernel_size=2, stride=2)
            old_out = self.relu(self.convs_img[-(resolution - 2)](old_out))

            out = self.resnet[-ind](out)

            # add
            new_out = (1 - alpha) * old_out + alpha * out

            for layer in range(-ind + 1, 0):
                new_out = self.resnet[layer](new_out)

        new_out = new_out.view(batch_size, self.nf0*self.s0*self.s0)
        new_out = self.fc(new_out)

        return new_out
