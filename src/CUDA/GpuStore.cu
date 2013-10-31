#include <cuda_runtime.h>
#include <stdio.h>
#include "../Store/storeElement.h"
#include "../Store/infoElement.h"
#include "../Store/LoggerHelper.h"

#define DEBUG 1
#define MB_SIZE 1048576
#define CHECK_CUDA_ERR(sth) { gpuAssert((sth), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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

		#if DEBUG
			fprintf(stdout,
					"Free memory: %.2f MB Total memory: %.2f MB File[%s] Line[%d]\n",
					(float)*freeMemory/MB_SIZE,
					(float)*totalMemory/MB_SIZE,
					__FILE__,
					__LINE__);
		#endif
	}

	int gpuAllocateMainArray(size_t size, void** array)
	{
		size_t freeMemory, totalMemory;

		#if DEBUG
		{
			fprintf(stdout,
					"Attempt to allocate %.2f MB memory on GPU File[%s] Line[%d]\n",
					(float)size/MB_SIZE,
					__FILE__,
					__LINE__);
			gpuGetMemoryUsage(&freeMemory, &totalMemory);
		}
		#endif

		cudaError_t result = cudaSuccess;

		if(totalMemory <= size)
		{
			result = cudaErrorMemoryAllocation;
			fprintf(stderr, "Size of memory to allocate is bigger than total gpu memory\n");
			return result;
		}

		result = cudaMalloc((void**)array, size);

		#if DEBUG
		{
			if(result == cudaSuccess)
				fprintf(stdout, "Main gpu array allocated successfully\n");
			else
				fprintf(stderr, "Main gpu array allocation failed! - %s\n", cudaGetErrorString(result));
			gpuGetMemoryUsage(&freeMemory, &totalMemory);
		}
		#endif

		return result;
	}

	void gpuFreeMemory(void* devPtr)
	{
		size_t freeMemory, totalMemory;
		#if DEBUG
			fprintf(stdout, "Releasing gpu pointer\n");
		#endif
		CHECK_CUDA_ERR( cudaFree(devPtr) );
		gpuGetMemoryUsage(&freeMemory, &totalMemory);
	}

}	/* extern "C" */
