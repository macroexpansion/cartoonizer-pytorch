import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def resize_crop(image: np.ndarray):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def calculate_padding(in_dim, out_dim, kernel_size=3, stride=1, dilation=1):
    w_padding = ((out_dim[0] - 1) * stride + kernel_size + (kernel_size - 1) *
                 (dilation - 1) - in_dim[0]) / 2
    h_padding = ((out_dim[1] - 1) * stride + kernel_size + (kernel_size - 1) *
                 (dilation - 1) - in_dim[1]) / 2
    print(int(w_padding), int(h_padding))
    return int(w_padding), int(h_padding)


def tf_padding(x, weights_size, stride=2, mode='same'):
    if mode == 'same':
        if x.shape[2] % stride == 0:
            pad = max(weights_size[-1] - stride, 0)
        else:
            pad = max(weights_size[-1] - (x.shape[2] % stride), 0)

        if pad % 2 == 0:
            pad_val = pad // 2
            padding = (pad_val, pad_val, pad_val, pad_val)
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end) # left, right, top, bottom

        return F.pad(x, padding, 'constant', 0)
    else:
        raise NotImplementedError(f'Not implement for padding mode: {mode}')
