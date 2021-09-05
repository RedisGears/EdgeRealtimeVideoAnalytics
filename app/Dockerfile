FROM python:3.8.10

WORKDIR /app
ADD . /app

RUN set -ex; \
    pip install -r requirements.txt;

ENTRYPOINT [ "python3" ]
