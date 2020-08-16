import cv2
import numpy as np
import torch

from net import UNet
from guided_filter import guided_filter
from utils import resize_crop


def load_model(device):
    net = UNet()
    net = net.to(device)
    net.load_state_dict(torch.load('weights.pt', map_location=device))
    net.eval()
    return net


def infer(image_path='test_images/food6.jpg', result_path='result.jpg'):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        net = load_model(device)

        image = cv2.imread(image_path)
        image = resize_crop(image)
        
        image = torch.from_numpy(image).float().permute(2, 0, 1).to(device) / 127.5 - 1
        image = image.unsqueeze(0)

        out = net(image)
        
        out = guided_filter(x=image, y=out, r=1, eps=5e-3)
        
        out = out.squeeze(0)
        out = (out + 1) * 127.5
        out = out.cpu().detach().permute(1, 2, 0).numpy()
        out = np.clip(out, 0, 255).astype(np.uint8)
        cv2.imwrite(result_path, out)


if __name__ == '__main__':
    import os

    res_folder = './results2'
    if not os.path.exists(res_folder):
        os.mkdir(res_folder)

    # images = [os.path.join('test_images', i) for i in os.listdir('test_images')]
    images = ['./test_images/party7.jpg']
    for image in images:
        infer(image_path=image, result_path=os.path.join(res_folder, image.split('/')[-1]))

