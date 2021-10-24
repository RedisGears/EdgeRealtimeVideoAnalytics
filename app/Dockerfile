FROM python:3.7.3

WORKDIR /app
ADD . /app

RUN pip install --upgrade pip
RUN set -ex; \
    pip install -r requirements.txt;

ENTRYPOINT [ "python3" ]
