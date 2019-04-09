FROM redislabs/redisedge:latest
# TODO: freeze version

ENV DEPS "libglib2.0-0 libsm6 libxrender1 libxext6 python-pip"
ENV PIP_LIBS "numpy pillow opencv-python"
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends $DEPS; \
    pip install -t /usr/local/lib/python2.7/site-packages $PIP_LIBS;
