import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import tf_padding

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def box_filter(x, r):
    channels = x.size(1)
    k_size = int(2 * r + 1)
    weight = 1. / (k_size**2)

    box_kernel = weight * torch.ones(channels, 1, k_size, k_size, dtype=torch.float32)
    depthwise_conv2d = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k_size, groups=channels, bias=False)
    depthwise_conv2d.weight = torch.nn.Parameter(box_kernel)
    depthwise_conv2d = depthwise_conv2d.to(device)
    x_pad = tf_padding(x, depthwise_conv2d.weight.size(), stride=1, mode='same')
    output = depthwise_conv2d(x_pad.to(device))

    return output


def guided_filter(x, y, r=1, eps=5e-3):
    _, c, h, w = x.size()

    k = torch.ones(1, 1, h, w, dtype=x.dtype)
    N = box_filter(k, r)

    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output

if __name__ == '__main__':
    x = torch.ones(1,3,16,12, dtype=torch.float32)
    y = guided_filter(x, x)
    print(y.size())