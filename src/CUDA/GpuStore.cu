#include <cuda_runtime.h>
#include <stdio.h>
#include "../Store/storeElement.h"
#include "../Store/infoElement.h"
#include "../Helpers/Logger.h"

#define DEBUG 1
#define MB_SIZE 1048576
#define CHECK_CUDA_ERR(sth) { gpuAssert((sth), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
	  LOG4CPLUS_ERROR(Logger::getRoot(), LOG4CPLUS_TEXT("Error...."));
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

Logger logger = Logger::getRoot();

extern "C"
{
	int gpuGetCudaDevicesCount()
	{
		int count = 0;
		CHECK_CUDA_ERR( cudaGetDeviceCount(&count) );
		return count;
	}

	void gpuGetMemoryUsage(size_t* freeMemory, size_t* totalMemory)
	{
		CHECK_CUDA_ERR( cudaMemGetInfo(freeMemory, totalMemory) );
		LOG4CPLUS_INFO(logger, "Free memory: " << (float)*freeMemory/MB_SIZE << " MB Total memory: " << (float)*totalMemory/MB_SIZE << " MB");
	}

	int gpuAllocateMainArray(size_t size, void** array)
	{
		size_t freeMemory, totalMemory;
		LOG4CPLUS_INFO(logger, "Attempt to allocate " << (float)size/MB_SIZE << " MB memory on GPU ");

		gpuGetMemoryUsage(&freeMemory, &totalMemory);

		cudaError_t result = cudaSuccess;

		if(totalMemory <= size)
		{
			result = cudaErrorMemoryAllocation;
			LOG4CPLUS_ERROR(logger, LOG4CPLUS_TEXT("Size of memory to allocate is bigger than total gpu memory\n"));
			return result;
		}

		result = cudaMalloc((void**)array, size);

		if(result == cudaSuccess)
			LOG4CPLUS_INFO(logger, LOG4CPLUS_TEXT("Main gpu array allocated successfully"));
		else
			LOG4CPLUS_ERROR(logger, "Main gpu array allocation failed! - %s\n" << cudaGetErrorString(result));
		gpuGetMemoryUsage(&freeMemory, &totalMemory);

		return result;
	}

	void gpuFreeMemory(void* devPtr)
	{
		size_t freeMemory, totalMemory;
		CHECK_CUDA_ERR( cudaFree(devPtr) );
		gpuGetMemoryUsage(&freeMemory, &totalMemory);
	}

}	/* extern "C" */
