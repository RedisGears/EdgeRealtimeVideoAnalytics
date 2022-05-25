FROM python:3.7.3

WORKDIR /app
ADD . /app

RUN set -ex; \
    pip install --upgrade pip; \
    pip install -r requirements.txt;

ENTRYPOINT [ "python3" ]
