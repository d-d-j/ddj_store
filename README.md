1. Clone repo or import it with Eclipse
2. Install required libs

		sudo apt-get install libstdc++6-4.6-dev libboost-all-dev
3. Download log4cplus-1.1.2 from SourceForge and follow instructions in INSTALL file
4. export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
5. build application with `make all`
6. Run program


### Common Problems

1. ` error while loading shared libraries: libcudart.so.5.5:`
There are some environmental variables needed for CUDA:

		export PATH=/usr/local/cuda-5.5/bin:$PATH
		export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/usr/local/lib:$LD_LIBRARY_PATH
