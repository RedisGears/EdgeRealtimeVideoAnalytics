FROM python:3.10.0b2

WORKDIR /app
ADD . /app
RUN apt-get update -qyy
RUN apt-get install -y  python3-opencv
RUN apt-get remove -y python3-opencv

RUN pip install --upgrade pip
RUN set -ex; \
    pip install -r requirements.txt;
RUN rm -rf ~/.cache/pip /var/cache/apt/

ENTRYPOINT [ "python3" ]
