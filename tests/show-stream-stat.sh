#!/bin/bash

if [[ -z $DOCKER_HOST ]]; then
	host_arg=""
else
	host_arg="-h $(echo $DOCKER_HOST|cut -d: -f1)"
fi

stream_size() {
	echo $(redis-cli $host_arg xlen $1|cat)
}

ts_get() {
	echo $(redis-cli $host_arg ts.get $1|cat)
}

echo camera: $(stream_size camera:0:yolo)
echo ts/camera: $(ts_get camera:0:out_fps)
echo people: $(ts_get camera:0:people)
echo log: $(stream_size log)
