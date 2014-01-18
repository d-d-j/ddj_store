#DDJ STORE

### Requirements

1. gcc 4.7.3 or higher
2. CUDA 5.5

### Installation

1. Clone repo or import it with Eclipse
2. Install required libs. `./install.sh`
3. Setup CUDA required variables. Add following lines to `/etc/profile`

		export PATH=/usr/local/cuda-5.5/bin:$PATH
		export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/usr/local/lib:$LD_LIBRARY_PATH

4. build application with `make all`
7. Copy configuration files `config.ini` and `ddj_logger.prop` from their samples and edit as you wish
5. Run all tests `make test`
6. Run program `make run`


### Common Problems

1. ` error while loading shared libraries: libcudart.so.5.5:` or `nvcc fatal   : Path to libdevice library not specified`
There are some environmental variables needed for CUDA:

		export PATH=/usr/local/cuda-5.5/bin:$PATH
		export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:/usr/local/lib:$LD_LIBRARY_PATH



