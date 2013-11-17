#ifndef GPUSTORE_CUH_
#define GPUSTORE_CUH_

#include "../Store/StoreIncludes.h"

extern "C"
{
	void gpuGetMemoryUsage(size_t* freeMemory, size_t* totalMemory);

	int gpuAllocateMainArray(size_t size, void** array);

	void gpuFreeMemory(void* devPtr);

	int gpuGetCudaDevicesCount();

	int gpuAppendToMainStore(void* data, size_t dataSize, void* mainStorePointer, ullint mainStoreOffset);

	int gpuGetCudaDevicesCountAndPrint();

	bool gpuCheckCudaDevice(int n);
} /* extern "C" */
#endif /* GPUSTORE_CUH_ */
