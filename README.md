1. Clone repo or import it with Eclipse
2. Install required libs

		sudo apt-get install libstdc++6-4.6-dev libboost-all-dev
3. Download log4cplus-1.1.2 from SourceForge and follow instructions in INSTALL file
4. export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

5. Download boost threadpool from https://github.com/AlexMarlo/boost-threadpool/blob/master/boost/threadpool/task_adaptors.hpp
6. copy /boost/threadpool to your boost includes path

7. build application with `make all`
8. Run program


### Common Problems

1. ` error while loading shared libraries: libcudart.so.5.5:`
There are some environmental variables needed for CUDA:

		export PATH=/usr/local/cuda-5.5/bin:$PATH
		export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/usr/local/lib:$LD_LIBRARY_PATH
