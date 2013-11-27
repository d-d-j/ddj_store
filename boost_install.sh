#!/bin/bash

#Update repositories list
sudo apt-get update

#Completely remove boost
sudo apt-get -y --purge remove libboost-all-dev libboost-doc libboost-dev
sudo rm -fv /usr/lib/libboost_*

#Install required packages
sudo apt-get -y install build-essential g++ python-dev autotools-dev libicu-dev libbz2-dev

#Download boost
cd /tmp
wget http://downloads.sourceforge.net/project/boost/boost/1.54.0/boost_1_54_0.tar.gz
tar -zxvf boost_1_54_0.tar.gz
rm -fv boost_1_54_0.tar.gz
cd boost_1_54_0

#Compile and install
./bootstrap.sh --prefix=/usr/local
cpuCores=`cat /proc/cpuinfo | grep "cpu cores" | uniq | awk '{print $NF}'`
echo "Available CPU cores: "$cpuCores
sudo ./b2 --with=all -j $cpuCores install

#Install boost-thread-pool
wget https://raw.github.com/AlexMarlo/boost-threadpool/master/boost/threadpool/task_adaptors.hpp
sudo mv task_adaptors.hpp /usr/local/include/boost/task_adaptors.hpp


