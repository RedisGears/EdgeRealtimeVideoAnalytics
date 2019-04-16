FROM redislabs/redisedge:latest
# TODO: freeze version

ENV DEPS "libglib2.0-0 libsm6 libxrender1 libxext6 python-pip"
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends $DEPS;

ADD ./requirements.txt /tmp/requirements.txt
RUN set -ex; \
    pip install -t /usr/local/lib/python2.7/site-packages -r /tmp/requirements.txt;
# TODO: install to virtual env once https://github.com/RedisLabsModules/RedisGears/pull/85 is merged
