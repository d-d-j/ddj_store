#!/bin/bash

#Update repositories list
sudo apt-get update

#Completely remove boost
sudo apt-get -y --purge remove libboost-all-dev libboost-doc libboost-dev
sudo rm -f /usr/lib/libboost_*

#Install required packages
sudo apt-get -y install build-essential g++ python-dev autotools-dev libicu-dev libbz2-dev

#Download boost
cd /tmp
wget http://downloads.sourceforge.net/project/boost/boost/1.54.0/boost_1_54_0.tar.gz
tar -zxf boost_1_54_0.tar.gz
rm -f boost_1_54_0.tar.gz
cd boost_1_54_0

#Compile and install (this may not work on multi processors machines)
./bootstrap.sh --prefix=/usr/local
cpuCores=`cat /proc/cpuinfo | grep "cpu cores" | uniq | awk '{print $4}'`
echo "Available CPU cores: "$cpuCores
sudo ./b2 --with=all -j 24 install
cd ..
rm -rf boost_1_54_0

#Install boost-thread-pool
git clone https://github.com/AlexMarlo/boost-threadpool.git --depth 1
sudo mv boost-threadpool/boost/threadpool /usr/local/include/boost/
sudo mv boost-threadpool/boost/threadpool.hpp /usr/local/include/boost/
rm -rf boost-threadpool


