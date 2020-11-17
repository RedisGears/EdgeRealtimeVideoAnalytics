FROM redislabs/redismod:latest

ENV DEPS "python python3-pip python3-setuptools libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1"
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends $DEPS;

RUN set -ex; \
    pip3 install -U pipenv;

ADD ./requirements.txt /tmp/requirements.txt
RUN set -ex; \
    cd /var/opt/redislabs/modules/rg/python3; \
    pipenv run pip3 install -r /tmp/requirements.txt;

ADD ./redisedge.conf /usr/local/etc/redisedge.conf

CMD ["/usr/local/etc/redisedge.conf"]
