FROM python:3.7.3

ENV PIP_LIBS "numpy pillow opencv-python redis flask"
RUN set -ex; \
    pip install $PIP_LIBS;

WORKDIR /src
ADD . /src