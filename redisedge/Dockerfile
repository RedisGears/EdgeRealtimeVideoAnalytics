FROM redislabs/redisedge:latest

WORKDIR /opt/redislabs/lib/modules/python3
ADD ./requirements.txt /tmp/requirements.txt
ADD ./redisedge.conf /usr/local/etc/redisedge.conf

RUN set -ex; \
    apt-get --allow-releaseinfo-change update; \
    apt-get update; \
    apt-get install -y --no-install-recommends python python3-pip python3-setuptools libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1

RUN pip3 install --upgrade pip
RUN pip3 install -U pipenv

RUN pipenv run pip3 install -r /tmp/requirements.txt;


CMD ["/usr/local/etc/redisedge.conf"]
