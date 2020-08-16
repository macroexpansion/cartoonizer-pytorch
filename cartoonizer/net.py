import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import tf_padding


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super(UNet, self).__init__()
        self.encoder0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
                                nn.LeakyReLU(0.2))
        self.encoder1 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, stride=2),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1),
                                nn.LeakyReLU(0.2))
        self.encoder2 = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=0, stride=2),
                                nn.LeakyReLU(0.2), 
                                nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=3, padding=1),
                                nn.LeakyReLU(0.2))

        self.resblock0 = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, padding=1))
        self.resblock1 = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, padding=1))
        self.resblock2 = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, padding=1))
        self.resblock3 = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, padding=1))

        self.decoder0 = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels * 2, kernel_size=3, padding=1),
                                nn.LeakyReLU(0.2))
        self.decoder1 = nn.Sequential(nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, padding=1),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                                nn.LeakyReLU(0.2))
        self.decoder2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                nn.LeakyReLU(0.2),
                                nn.Conv2d(out_channels, 3, kernel_size=7, padding=3))

    def forward(self, input):
        x0 = self.encoder0(input)

        x0_tf_padding = tf_padding(x0, weights_size=self.encoder1[0].weight.size(), stride=2)
        x1 = self.encoder1(x0_tf_padding)
        x1_tf_padding = tf_padding(x1, weights_size=self.encoder2[0].weight.size(), stride=2)
        x2 = self.encoder2(x1_tf_padding)

        r0 = self.resblock0(x2)
        residual = r0 + x2
        r1 = self.resblock1(residual)
        residual = r1 + residual
        r2 = self.resblock2(residual)
        residual = r2 + residual
        r3 = self.resblock3(residual)
        residual = r3 + residual

        r3 = self.decoder0(residual)

        h1, w1 = r3.size(2), r3.size(3)
        x3 = F.interpolate(r3, (h1*2, w1*2), mode='bilinear', align_corners=True)
        x3 = self.decoder1(x3 + x1)

        h2, w2 = x3.size(2), x3.size(3)
        x4 = F.interpolate(x3, (h2*2, w2*2), mode='bilinear', align_corners=True)
        x4 = self.decoder2(x4 + x0)

        return x4