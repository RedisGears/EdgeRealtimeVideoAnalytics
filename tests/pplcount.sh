#!/bin/bash

# [[ $VERBOSE == 1 ]] && set -x

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"  
cd $HERE

if [[ -z $DOCKER_HOST ]]; then
	host_arg=""
else
	host_arg="-h $(echo $DOCKER_HOST|cut -d: -f1)"
fi

PROJECT=pplcount
DOCKER_LOG=/tmp/pplcount.log
# --compatibility 
# --no-ansi 
SPEC="-p $PROJECT -f docker-compose.yaml"
[[ $REBUILD == 1 ]] && BUILD_ARG="--build"

rm -f $DOCKER_LOG

start() {
	if [[ $VERBOSE == 1 ]]; then
		docker-compose $SPEC up $BUILD_ARG -d
	else
		docker-compose $SPEC up $BUILD_ARG -d >> $DOCKER_LOG 2>&1
	fi
	BUILD_ARG=''
}

stop() {
	RMI_ARG=""
	[[ $ALL == 1 ]] && RMI_ARG="--rmi all"
	if [[ $VERBOSE == 1 ]]; then
		$HERE/show-stream-stat.sh
		$HERE/show-log.sh
		docker-compose $SPEC down $RMI_ARG -v --remove-orphans
	else
		#  --rmi local
		docker-compose $SPEC down $RMI_ARG -v --remove-orphans >> $DOCKER_LOG 2>&1
	fi
}

count() {
	echo $(redis-cli $host_arg TS.GET camera:0:people | cat)
}

build() {
	if [[ $VERBOSE == 1 ]]; then
		docker-compose $SPEC build
	else
		docker-compose $SPEC build >> $DOCKER_LOG 2>&1
	fi
}

clean() {
	ALL=1 stop
}

show_logs() {
	docker-compose $SPEC logs $*
	./show-log.sh
	./show-stream-stat.sh
}

run_test() {
	echo "Counting people ..."
	start
	for ((i = 0; i < 3; i++)); do
		sleep 10
		[[ $VERBOSE == 1 ]] && ./show-stream-stat.sh
		ppl_count=$(count)
		if [[ $ppl_count > 0 ]]; then
			break
		fi
	done
	
	ppl_count=$(count)
	if [[ $VERBOSE == 1 ]]; then
		echo "ppl_count=$ppl_count"
		show_logs
	else
		echo "ppl_count=$ppl_count" >> $DOCKER_LOG
		show_logs >> $DOCKER_LOG
	fi
	stop
}

help() {
	echo "[VERBOSE=0|1] [REBUILD=0|1] $0 [start|stop|build|clean|count|logs|help]"
}

cmd=$1
shift
if [[ $cmd == help ]]; then
	help
	exit 0
elif [[ $cmd == start ]]; then
	start
elif [[ $cmd == stop ]]; then
	stop
elif [[ $cmd == build ]]; then
	build
elif [[ $cmd == clean ]]; then
	clean
elif [[ $cmd == count ]]; then
	echo $(count)
	exit 0
elif [[ $cmd == logs ]]; then
	show_logs $*
elif [[ $cmd == "" ]]; then
    run_test
	if [[ -z $ppl_count || $ppl_count == 0 ]]; then
		echo "pplcount: FAIL"
		exit 1
	fi
	echo "pplcount: OK"

else
	help
	exit 0
fi
exit 0
