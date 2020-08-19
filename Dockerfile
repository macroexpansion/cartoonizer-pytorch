# syntax=docker/dockerfile:experimental
ARG BASE_IMAGE=ubuntu:18.04

FROM ${BASE_IMAGE} AS compile-image

ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    python3-dev \
    python3-distutils \
    python3-venv \
    openjdk-11-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN python3 -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# This is only useful for cuda env 
# RUN export USE_CUDA=1

ADD requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir 

# Final image for production
FROM ${BASE_IMAGE} AS runtime-image

ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3 \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp

COPY --from=compile-image /home/venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh \
    && chown -R model-server /home/model-server

COPY config.properties /home/model-server/config.properties
COPY model_store /home/model-server/model_store
# RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store
RUN chown -R model-server /home/model-server/model-store

EXPOSE 8080 8081

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]