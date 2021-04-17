#!/bin/bash

if [[ -z $DOCKER_HOST ]]; then
	host_arg=""
else
	host_arg="-h $(echo $DOCKER_HOST|cut -d: -f1)"
fi

redis-cli $host_arg xrange log - + | ./fold.py 16:-:50
