#!/bin/bash

for bufferSize in 32 64 128 256 512 1024 2048 4096
do
for poolSize in 2
do
   > config.ini
   echo "MB_SIZE_IN_BYTES = 1048576
	MAIN_STORE_SIZE = 536870912
	GPU_MEMORY_ALLOC_ATTEMPTS = 6
	STREAMS_NUM_QUERY = $poolSize
	STREAMS_NUM_UPLOAD = $poolSize
	SIMUL_QUERY_COUNT = $poolSize
	STORE_BUFFER_CAPACITY = $bufferSize
	THREAD_POOL_SIZE = $poolSize
	MASTER_IP_ADDRESS = 127.0.0.1
	MASTER_LOGIN_PORT = 8080" > config.ini
	./DDJ_Store --performance
done
done
