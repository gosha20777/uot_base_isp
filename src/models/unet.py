import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np


class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """
    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.InstanceNorm2d(mid_channels),
            PixelNormLayer(),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.InstanceNorm2d(out_channels),
            PixelNormLayer(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks=4, base_factor=32, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_factor = base_factor

        self.inc = DoubleConv(n_channels, base_factor)
        
        self.down1 = Down(base_factor, 2 * base_factor)
        self.down2 = Down(2 * base_factor, 4 * base_factor)
        self.down3 = Down(4 * base_factor, 8 * base_factor)
        
        factor = 2 if bilinear else 1
        
        self.down4 = Down(8 * base_factor, 16 * base_factor // factor)
        self.up1 = Up(16 * base_factor, 8 * base_factor // factor, bilinear)
        self.up2 = Up(8 * base_factor, 4 * base_factor // factor, bilinear)
        self.up3 = Up(4 * base_factor, 2 * base_factor // factor, bilinear)
        self.up4 = Up(2 * base_factor, base_factor, bilinear)
        
        self.outc = OutConv(base_factor, n_classes)

#         down_blocks = []
#         up_blocks = []
#         for i in range(n_blocks):
#             if i != n_blocks - 1:
#                 down_blocks.append(Down(2 ** i * base_factor, 2 ** (i + 1) * base_factor))
#             else:
#                 down_blocks.append(Down(2 ** i * base_factor, 2 ** (i + 1) * base_factor // factor))
            
#             if i == 0:
#                 up_blocks.append(Up(2 ** (i + 1) * base_factor, 2 ** i * base_factor, bilinear))
#             else:
#                 up_blocks.append(Up(2 ** (i + 1) * base_factor, 2 ** i * base_factor // factor, bilinear))
#         self.down_blocks = down_blocks
#         self.up_blocks = up_blocks
            

    def forward(self, x):
        x1 = self.inc(x)
#         xs = [self.inc(x)]
#         for layer in self.down_blocks:
#             xs.append(layer(xs[-1]))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

#         for i in range(self.n_blocks):
#             layer = self.up_blocks[self.n_blocks - i - 1]
#             if i == 0:
#                 x = layer(xs[-1], xs[-2])
#             else:
#                 x = layer(x, xs[-i - 2])
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet_Progressive(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks=4, base_factor=32, bilinear=True):
        super(UNet_Progressive, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_factor = base_factor
        self.n_blocks = n_blocks

        factor = 2 if bilinear else 1

        # self.inc = DoubleConv(n_channels, base_factor)
        self.incs = nn.ModuleList([DoubleConv(n_channels, 2 ** i * base_factor) for i in range(n_blocks)])

        self.outs = nn.ModuleList([OutConv(base_factor, n_channels)] +\
            [OutConv(2 ** i * base_factor // factor, n_channels) for i in range(1, n_blocks)])


        down_blocks = []
        up_blocks = []
        for i in range(n_blocks):
            if i != n_blocks - 1:
                down_blocks.append(Down(2 ** i * base_factor, 2 ** (i + 1) * base_factor))
            else:
                down_blocks.append(Down(2 ** i * base_factor, 2 ** (i + 1) * base_factor // factor))
            
            if i == 0:
                up_blocks.append(Up(2 ** (i + 1) * base_factor, 2 ** i * base_factor, bilinear))
            else:
                up_blocks.append(Up(2 ** (i + 1) * base_factor, 2 ** i * base_factor // factor, bilinear))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        
#         self.alpha = 0.5

    def forward(self, x, alpha):

        resolution = int(np.log2(x.size(2))) 
        xs = [self.incs[-(resolution-1)](x)]

        if resolution == 2: # for 4x4
            xs.append(self.down_blocks[-1](xs[-1]))
        
            x_up = self.up_blocks[-1](xs[-1], xs[-2])
            
            logits = self.outs[-1](x_up)

        else: # for 8x8 and up
            pool_x = F.avg_pool2d(x, kernel_size=2, stride=2)
            pool_x = self.incs[-(resolution-2)](pool_x)

            xs.append(self.down_blocks[-(resolution-1)](xs[-1]))

            new_x = (1 - alpha) * pool_x + alpha * xs[-1]

            for ind in range(-resolution + 2, 0):
                new_x = self.down_blocks[ind](new_x)
                xs.append(new_x)

            for i in range(resolution - 2): # number of down blocks + 1 (input_conv)
                layer = self.up_blocks[- i - 1]
                if i == 0:
                    x = layer(xs[-1], xs[-2])
                else:
                    x = layer(x, xs[-i - 2])

            new_x = self.up_blocks[-resolution + 1](x, xs[-resolution])
            new_x = self.outs[-resolution+1](new_x)
            
            # to_rgb + upscale
            old_x = self.outs[-resolution+2](x)
            old_x = F.upsample(old_x, scale_factor=2)

            logits = (1 - alpha) * old_x + alpha * new_x

        return logits
