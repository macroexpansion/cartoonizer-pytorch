# Cartoonizer Pytorch
This Pytorch implementation contains only inference code based on Tensorflow implementation from [White Box Cartoonization](https://github.com/SystemErrorWang/White-box-Cartoonization)

## Usage
```
curl -o https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl https://cartoonize-pvowi6uvjq-as.a.run.app/predictions/cartoonize -T kitten.jpg -o res.jpg
```

## Installation
Run with torchserve and Docker

1. Install requirements:
```
pip install -r requirements.txt
```
2. Create a torchserve model archiver:
```
sh create_mode_archiver.sh
```
3. Run with torchserve
```
sh start.sh
```

- To stop torchserve
```
sh stop.sh
```
4. Inference
```
curl -o https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl http://127.0.0.1:8080/predictions/cartoonize -T kitten.jpg -o res.jpg
```
5. (Optional) Build and deploy with docker

- Build
```
DOCKER_BUILDKIT=1 docker build -t cartoonize .
```

- Run
```
docker run --name cartoonize --rm -it -p 8080:8080 -p 8081:8081 cartoonize
```

- Stop
```
docker stop cartoonize
```

## License
- Copyright (C) [Xinrui Wang](https://github.com/SystemErrorWang) All rights reserved. Licensed under the CC BY-NC-SA 4.0
- license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
- Commercial application is prohibited, please remain this license if you clone this repo
