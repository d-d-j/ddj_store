1. Clone repo or import it with Eclipse
2. Install required libs
	
		sudo apt-get install libstdc++6-4.6-dev libboost-all-dev

3. build application with `make all`
4. Run program

There are some environmental variables needed for CUDA:
	1. export PATH=/usr/local/cuda-5.5/bin:$PATH
	2. export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:$LD_LIBRARY_PATH

