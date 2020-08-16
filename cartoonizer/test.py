from handler import handle
import io
from PIL import Image


class Context:
    def __init__(self):
        self.manifest = {
            'model': {
                'serializedFile': 'weights.pt'
            }
        }
        self.system_properties = {
            'gpu_id': '0'
        }


context = Context()

f = open("test_images/party7.jpg", "rb")
image = Image.open(f)
byte_array = io.BytesIO()
image.save(byte_array, format='jpeg')
byte_array = byte_array.getvalue()

res = handle([{'data': byte_array}], context)
img = Image.open(io.BytesIO(res[0]))
img.save('test.jpg')