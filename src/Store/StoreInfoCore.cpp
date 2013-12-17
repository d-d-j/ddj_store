#include "StoreInfoCore.h"

namespace ddj
{
namespace store
{
size_t StoreInfoCore::GetNodeInfo(void** result)
{
	size_t mbSize = this->_config->GetIntValue("MB_SIZE_IN_BYTES");

	size_t gpuMemFree, gpuMemTotal;
	int memTotal, memFree;

	_cudaCommons.GetMemoryCount(&gpuMemFree, &gpuMemTotal);
	LOG4CPLUS_INFO_FMT(this->_logger,
			"NODE INFO - free GPU memory => %f MB, total GPU memory => %f MB",
			(float ) gpuMemFree / mbSize,
			(float ) gpuMemTotal / mbSize);

// TODO: move getting ram info to Node instear of here
	GetRamInKB(&memTotal, &memFree);

	LOG4CPLUS_INFO_FMT(this->_logger,
			"NODE INFO - free RAM memory => %f MB, total RAM memory => %f MB",
			(float )memFree / 1024, (float )memTotal / 1024);


	StoreNodeInfo* nodeInfo = new StoreNodeInfo(memTotal, memFree, gpuMemTotal, gpuMemFree);

	*result = &nodeInfo;
	return sizeof(nodeInfo);
}

void StoreInfoCore::GetRamInKB(int* ramTotal, int* ramFree)
{
	*ramTotal = *ramFree = -1;
	FILE *meminfo = fopen("/proc/meminfo", "r");
	if (meminfo == NULL)
	{
		LOG4CPLUS_ERROR(this->_logger,
				LOG4CPLUS_TEXT("Unable to open meminfo"));
		return;
	}

	char line[256];

	while (fgets(line, sizeof(line), meminfo))
	{
		sscanf(line, "MemTotal: %d kB", ramTotal);
		if (sscanf(line, "MemFree: %d kB", ramFree) == 1)
		{
			continue;
		}
	}

	fclose(meminfo);
}

StoreInfoCore::StoreInfoCore()
{
	// TODO Auto-generated constructor stub

}
}
}
