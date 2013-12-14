#!/bin/bash

mkdir gtestInstallTemp
cd gtestInstallTemp

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    wget https://googletest.googlecode.com/files/gtest-1.7.0.zip
    unzip gtest-1.7.0.zip
    cd gtest-1.7.0
	mkdir mybuild
	cd mybuild
	cmake -DGTEST_HAS_PTHREAD=1 ../
	make
elif [[ "$OSTYPE" == "darwin"* ]]; then
	curl -O https://googletest.googlecode.com/files/gtest-1.7.0.zip
	unzip gtest-1.7.0.zip
	cd gtest-1.7.0
	mkdir mybuild
	cd mybuild
	cmake -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_CXX_FLAGS="-std=c++11 -stdlib=libc++ -U__STRICT_ANSI__" -DGTEST_HAS_PTHREAD=1 ../
	make
fi
sudo cp -r ../include/gtest /usr/local/include
sudo cp lib*.a /usr/local/lib
cd ../../../
rm -R gtestInstallTemp