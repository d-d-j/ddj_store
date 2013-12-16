#!/bin/bash

cd /tmp

wget http://sourceforge.net/projects/log4cplus/files/log4cplus-stable/1.1.2/log4cplus-1.1.2.tar.gz
tar -xf log4cplus-1.1.2.tar.gz
rm -fv log4cplus-1.1.2.tar.gz

cd log4cplus-1.1.2

./configure
make
sudo make install
make installcheck

rm -rf log4cplus-1.1.2