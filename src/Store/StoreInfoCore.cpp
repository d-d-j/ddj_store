#include "StoreInfoCore.h"

namespace ddj
{
namespace store
{
size_t StoreInfoCore::GetNodeInfo(storeNodeInfo** result)
{
	size_t gpuMemFree, gpuMemTotal;
	int memTotal, memFree;

	_cudaCommons.GetMemoryCount(&gpuMemFree, &gpuMemTotal);

	// TODO: move getting ram info to Node instead of here
	GetRamInKB(&memTotal, &memFree);

	(*result)->db_mem_free = _cudaController->GetMainMemoryOffset();
	(*result)->gpu_id =  _cudaController->GetCudaDeviceId();
	(*result)->mem_free = memFree;
	(*result)->mem_total = memTotal;
	(*result)->gpu_mem_free = gpuMemFree;
	(*result)->gpu_mem_total = gpuMemTotal;

	return sizeof(storeNodeInfo);
}

void StoreInfoCore::GetRamInKB(int* ramTotal, int* ramFree)
{
	// TODO: Implement for Mac OS X
	*ramTotal = *ramFree = -1;
	FILE *meminfo = fopen("/proc/meminfo", "r");
	if (meminfo == nullptr)
	{
		LOG4CPLUS_ERROR(this->_logger,
				LOG4CPLUS_TEXT("Unable to open meminfo"));
		return;
	}

	char line[256];

	while (fgets(line, sizeof(line), meminfo))
	{
		sscanf(line, "MemTotal: %10d kB", ramTotal);
		if (sscanf(line, "MemFree: %10d kB", ramFree) == 1)
		{
			continue;
		}
	}

	fclose(meminfo);
}

}
}
