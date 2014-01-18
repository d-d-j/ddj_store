#!/bin/bash

sudo apt-get install cmake

#Create temp directory for installation
mkdir gtestInstallTemp
cd gtestInstallTemp

#Download gtest and extract it
curl -O https://googletest.googlecode.com/files/gtest-1.7.0.zip
unzip gtest-1.7.0.zip
cd gtest-1.7.0
mkdir mybuild
cd mybuild

#Generate Makefile depending on OS
if [[ "$OSTYPE" == "linux-gnu" ]]; then
	cmake -DGTEST_HAS_PTHREAD=1 ../
elif [[ "$OSTYPE" == "darwin"* ]]; then
	cmake -DCMAKE_CXX_COMPILER="clang++" -DCMAKE_CXX_FLAGS="-std=c++11 -stdlib=libc++ -U__STRICT_ANSI__" -DGTEST_HAS_PTHREAD=1 ../
fi

#Build
make

#Install
sudo cp -r ../include/gtest /usr/local/include
sudo cp lib*.a /usr/local/lib

#Remove temp files
cd ../../../
rm -Rf gtestInstallTemp
