FROM redislabs/redisedge:latest

ENV DEPS "python python-pip libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1"
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends $DEPS;

RUN set -ex; \
    pip install pipenv;

ADD ./requirements.txt /tmp/requirements.txt
RUN set -ex; \
    cd /opt/redislabs/lib/modules/python3; \
    pipenv run pip install -r /tmp/requirements.txt;

ADD ./redisedge.conf /usr/local/etc/redisedge.conf

CMD ["/usr/local/etc/redisedge.conf"]
