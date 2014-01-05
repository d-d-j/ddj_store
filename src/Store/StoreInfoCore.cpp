#include "StoreInfoCore.h"

namespace ddj
{
namespace store
{
size_t StoreInfoCore::GetNodeInfo(storeNodeInfo** result)
{
	size_t gpuMemFree, gpuMemTotal;
	int memTotal, memFree, gpuId;

	gpuId = _cudaController->GetCudaDeviceId();
	_cudaCommons.GetMemoryCount(&gpuMemFree, &gpuMemTotal);

	// TODO: move getting ram info to Node instead of here
	GetRamInKB(&memTotal, &memFree);

	*result = new storeNodeInfo(gpuId, memTotal, memFree, gpuMemTotal, gpuMemFree, _cudaController->GetMainMemoryOffset());

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
