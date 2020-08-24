import requests
from PIL import Image
import io
import base64

url = 'http://127.0.0.1:8080/predictions/cartoonize'
# url = 'https://cartoonize-pvowi6uvjq-as.a.run.app/predictions/cartoonize'
path = '../kitten.jpg'

f = open(path, "rb")
image = Image.open(f)
byte_array = io.BytesIO()
image.save(byte_array, format='jpeg')
byte_array = byte_array.getvalue()

res = requests.post(url, data=byte_array)
imgdata = base64.b64decode(res.text)
filename = 'test.jpg'
with open(filename, 'wb') as f:
    f.write(imgdata)