import torch
import logging
import io
import numpy as np
import os
import cv2

from PIL import Image
from guided_filter import guided_filter
from net import UNet
from utils import resize_crop

from ts.torch_handler.base_handler import BaseHandler


class ModelHandler(BaseHandler):
    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 1
        self.initialized = False
        self.net = None
        self.device = None


    def initialize(self, context):
        self._context = context
        self.manifest = context.manifest
        properties = context.system_properties
        # model_dir = properties.get('model_dir')

        self.device = torch.device('cuda:' + str(properties.get('gpu_id')) if torch.cuda.is_available() else 'cpu')
        # self._batch_size = context.system_properties["batch_size"]
        serialized_file = self.manifest['model']['serializedFile']
        if not os.path.isfile(serialized_file):
            raise RuntimeError("Missing the model weights file")
        # model_pt_path = os.path.join(model_dir, serialized_file)
        self.net = UNet()
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(serialized_file, map_location=self.device))
        self.net.eval()

        self.initialized = True


    def _image_processing(self, image: np.ndarray):
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('save.jpg', image)
        image = resize_crop(image)
        image = torch.from_numpy(image).float().permute(2, 0, 1).to(self.device) / 127.5 - 1
        return image


    def preprocess(self, batch):
        # Take the input data and pre-process it make it inference ready
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        with torch.no_grad():
            images = []
            for row in batch:
                image = row.get("data") or row.get("body")
                image = Image.open(io.BytesIO(image))
                image = self._image_processing(image)
                images.append(image)

            return torch.stack(images)


    def inference(self, model_input: torch.Tensor):
        # Do some inference call to engine here and return output
        with torch.no_grad():
            output = self.net(model_input)
        return output


    def postprocess(self, model_input: torch.Tensor, inference_output: torch.Tensor):
        # Take output from network and post-process to desired format
        with torch.no_grad():
            out = guided_filter(x=model_input, y=inference_output, r=1, eps=5e-3)
            out = out.squeeze(0)
            out = (out + 1) * 127.5
            out = out.cpu().detach().permute(1, 2, 0).numpy()
            out = np.clip(out, 0, 255).astype(np.uint8)
            out = out[...,::-1] # invert order of last channel to convert BGR -> RGB

            byte_array = io.BytesIO()
            image = Image.fromarray(out)
            image.save(byte_array, format='jpeg')
            byte_array = byte_array.getvalue()
            
        result = [byte_array]

        return result


    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_input, model_out)


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)