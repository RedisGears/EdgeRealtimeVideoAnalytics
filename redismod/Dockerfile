FROM redislabs/redismod:edge
FROM redislabs/redismod:edge

ENV DEPS "python python3-pip python3-setuptools libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1"
WORKDIR /data
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends $DEPS;
