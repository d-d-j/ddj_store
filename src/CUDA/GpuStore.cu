#include <cuda_runtime.h>
#include <stdio.h>
#include "../Store/storeSettings.h"

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

	int gpuGetCudaDevicesCountAndPrint()
	{
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		cudaDeviceProp prop;
		int driverVersion = 0, runtimeVersion = 0;

		printf("[File:%s][Line:%d] ==> CUDA : Found %d CUDA devices: \n\n", __FILE__, __LINE__, nDevices);
		for (int i = 0; i < nDevices; i++)
		{
			cudaSetDevice(i);
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);
			cudaGetDeviceProperties(&prop, i);

			// Log device query
			printf("	Device Number: %d\n", i);
			printf("	Device name: %s\n", prop.name);
	        printf("	CUDA Capability Major/Minor version number:    %d.%d\n", prop.major, prop.minor);
	        printf("	CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
			printf("	Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
			printf("	Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
			printf("	Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		}
		return nDevices;
	}

	bool gpuCheckCudaDevice(int n)
	{
		int driverVersion = 0, runtimeVersion = 0;
		cudaDeviceProp prop;
		cudaSetDevice(n);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		cudaGetDeviceProperties(&prop, n);

		if(prop.major < MIN_CUDA_MAJOR_VER) return false;
		if(prop.minor < MIN_CUDA_MINOR_VER) return false;

		return true;
	}

}	/* extern "C" */
