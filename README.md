1. Clone repo or import it with Eclipse
2. Install required libs

		sudo apt-get install libstdc++6-4.6-dev libboost-all-dev
3. Download log4cplus-1.1.2 from SourceForge and follow instructions in INSTALL file

		wget http://sourceforge.net/projects/log4cplus/files/log4cplus-stable/1.1.2/log4cplus-1.1.2.tar.gz
		tar -xvf log4cplus-1.1.2.tar.gz
		rm -fv log4cplus-1.1.2.tar.gz
		cd log4cplus-1.1.2
		./configure
		make
		sudo make install
		make installcheck

4. Setup CUDA required variables

		export PATH=/usr/local/cuda-5.5/bin:$PATH
		export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/usr/local/lib:$LD_LIBRARY_PATH

5. Install boost using `boost_install.sh`

6. Install google test (GTEST) :
	- You must have CMAKE http://www.cmake.org/cmake/resources/software.html and PHYTON >= 2.7 http://www.python.org/download/ installed
	- Download and unpack gtest-1.7.0.zip from https://code.google.com/p/googletest/downloads/list
	- Open unpacked folder and open this dir in console
	- Type following commands:
		$> mkdir mybuild
		$> cd mybuild
	- Build everything with cmake:
		a) FOR LINUX $> cmake -DGTEST_HAS_PTHREAD=1 ../
		b) FOR OS X   $> cmake -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_CXX_FLAGS="-std=c++11 -stdlib=libc++ -U__STRICT_ANSI__" -DGTEST_HAS_PTHREAD=1 ../     
	- Copy files to your include and lib path:
		$>  cp -r ../include/gtest /usr/local/include
		$>  cp lib*.a /usr/local/lib

7. build application with `make all`
8. Run program


### Common Problems

1. ` error while loading shared libraries: libcudart.so.5.5:`
There are some environmental variables needed for CUDA:

		export PATH=/usr/local/cuda-5.5/bin:$PATH
		export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/usr/local/lib:$LD_LIBRARY_PATH
		
		

