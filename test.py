import requests
from PIL import Image
import io

# url = 'http://127.0.0.1:8080/predictions/cartoonize'
url = 'https://cartoonize-pvowi6uvjq-as.a.run.app/predictions/cartoonize'

path = 'cartoonizer/test_images/mountain4.jpg'
f = open(path, "rb")
image = Image.open(f)
byte_array = io.BytesIO()
image.save(byte_array, format='jpeg')
byte_array = byte_array.getvalue()

res = requests.post(url, data=byte_array)
img = Image.open(io.BytesIO(res.content))
img.save('test.jpg')
