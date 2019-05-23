#!/bin/bash

../redis/src/redis-server --loadmodule ../redisai/build/redisai.so --loadmodule ../redistimeseries/src/redistimeseries.so --loadmodule ../redisgears/redisgears.so 
