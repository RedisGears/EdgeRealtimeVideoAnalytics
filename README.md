# RedisEdge Realtime Video Analytics

This is an example of using Redis Streams, RedisGears, RedisAI and RedisTimeSeries for Realtime Video Analytics (i.e. counting people).

TBD: animated gif

## Overview

This project demonstrates a possible deployment of the RedisEdge stack that provides realtime analytics of video streams.

The following diagram depicts the system's parts.

![Overview](/overview.png)

1. A video stream producer adds a captured frame to a Redis Stream.
2. The new frame triggers the execution of a RedisGear that:
    1. Downsamples the frame rate of the input stream, if needed.
    2. Resizes the input frame to the model's requirements.
    3. Calls RedisAI to execute an object recognition model on the frame.
    4. Stores the model's outputs (i.e. people counted and their whereabouts inside the frame) in RedisTimeSeries and Redis Stream data structures.
3. A video web server renders the final image based on realtime data from Redis' Streams.
4. Time series are exported from Redis to Prometheus, enabling visualization with Grafana's dashboards.

### The RedisEdge Stack

The RedisEdge stack consists of a the latest Redis release and select RedisLabs modules intended to be used in Edge computing. For more information refer to [RedisEdge](https://github.com/RedisLabs/redis-edge-docker).

### YOLOv3

TBD

## How to get it

```
$ git clone https://github.com/RedisGears/EdgeRealtimeVideoAnalytics.git
$ cd EdgeRealtimeVideoAnalytics
# If you don't have it already, install https://git-lfs.github.com/
$ git lfs install; git lfs checkout
```

## How to run it locally

### The RedisEdge stack

Refer to the build/installation instructions of the following projects to set up a Redis server with the relevant Redis modules. This application's connections default to `redis://localhost:6379`.

* [Redis](https://redis.io)
* [RedisGears](https://oss.redislabs.com/redisgears/)
* [RedisTimeSeries](https://oss.redislabs.com/redistimeseries/)
* [RedisAI](https://oss.redislabs.com/redisai/)

Note that you'll also need to install the Pythonic [`requirements.txt`](/redisedge/requirements.txt) for the embedded RedisGears Python interpreter.

### (optional) Prometheus and Grafana

Refer to the build/installation instructions of the following projects to set up Prometheus, Grafana and the RedisTimeSeries adapter:

* Prometheus: [Installation](https://prometheus.io/), [config sample](/prometheus/config.yml)
* Grafana: [Installation](https://grafana.com/), [config sample](/grafana/config.ini), [datasource sample](/grafana/provisioning/datasources/prometheus.yaml), [dashboard samples](/grafana/dashboards/)
* [prometheus-redistimeseries-adapter](https://github.com/RedisTimeSeries/prometheus-redistimeseries-adapter)

## The application

The application is implemented in Python and consists of the following parts:

- [`init.py`](/app/init.py): this initializes Redis with the RedisAI model, RedisTimeSeries downsampling rules and the RedisGears gear.
- [`gear.py`](/app/gear.py): input for the init process that registers the gear.
- [`capture.py`](/app/capture.py): captures video stream frames from a webcam or image/video file and store it in a Redis Stream.
- [`server.py`](/app/server.py): a web server that serves a rendered image composed of the raw frame and the model's detections.

To run the application you'll need Python v3.6 or higher. Install the application's library dependencies with:

```sh
$ cd app
$ pip install -r requirements.txt
```

The application's parts are set up with default values that are intended to allow it to run "out of the box". For example, to run the capture process you only need to type:

```sh
$ python3 capture.py
```

This will run the capture process from device id 0.

However. Most default values can be overridden from command line - invoke the application's parts with the `--help` switch to learn of these.

## How to run it with Docker Compose

```sh
$ docker-compose up
```

Note that you can choose which containers to spin. For example, the following will start the project with all components except the app itself:

```sh
$ docker-compose up redisedge prometheus grafana prometheus-redistimeseries-adapter
```

## UI

The application's video web server should be at http://localhost:5000/video. The Docker Compose setup comes with a pre-provisioned Grafana server that should be at http://localhost:3000/ (admin/admin).

## Known issues, limitations and todos

* TBD

## License

https://www.youtube.com/watch?v=VqkMaIk6fKc
