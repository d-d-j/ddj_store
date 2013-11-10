#include <cuda_runtime.h>

extern "C"
{
	int gpuGetCudaDevicesCount()
	{
		int count = 0;
		cudaGetDeviceCount(&count);
		return count;
	}

	void gpuGetMemoryUsage(size_t* freeMemory, size_t* totalMemory)
	{
		cudaMemGetInfo(freeMemory, totalMemory);
	}

	int gpuAllocateMainArray(size_t size, void** array)
	{
		size_t freeMemory, totalMemory;

		gpuGetMemoryUsage(&freeMemory, &totalMemory);

		cudaError_t result = cudaSuccess;

		if(totalMemory <= size)
		{
			result = cudaErrorMemoryAllocation;
			return result;
		}

		result = cudaMalloc((void**)array, size);

		gpuGetMemoryUsage(&freeMemory, &totalMemory);

		return result;
	}

	void gpuFreeMemory(void* devPtr)
	{
		size_t freeMemory, totalMemory;
		cudaFree(devPtr);
		gpuGetMemoryUsage(&freeMemory, &totalMemory);
	}

}	/* extern "C" */
