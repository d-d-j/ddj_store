#!/bin/bash
#Install add-apt-repository and add test ubuntu repo
sudo apt-get install python-software-properties
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
#Install gcc 4.7
sudo apt-get install gcc-4.7 g++-4.7 c++-4.7 -y
#Replace previous version with installed 4.7
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.7 30
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.7 30
sudo update-alternatives --config gcc
sudo update-alternatives --config g++