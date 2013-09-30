#ifndef GPUSTORE_CUH_
#define GPUSTORE_CUH_

extern "C"
{
	void gpuGetMemoryUsage(size_t* freeMemory, size_t* totalMemory);

	int gpuAllocateMainArray(size_t size, void** array);

	void gpuFreeMemory(void* devPtr);
} /* extern "C" */
#endif /* GPUSTORE_CUH_ */
