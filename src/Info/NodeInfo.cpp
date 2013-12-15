/*
 * NodeInfo.cpp
 *
 *  Created on: Dec 15, 2013
 *      Author: dud
 */

#include "NodeInfo.h"

namespace ddj
{
namespace store
{
void NodeInfo::FillNodeInfo()
{
	size_t mbSize = this->_config->GetIntValue("MB_SIZE_IN_BYTES");

	_cudaCommons.GetMemoryCount(&this->gpuMemFree, &this->gpuMemTotal);
	LOG4CPLUS_INFO_FMT(this->_logger,
			"NODE INFO - free GPU memory => %f MB, total GPU memory => %f MB",
			(float) this->gpuMemFree / mbSize, (float)this->gpuMemTotal / mbSize);

	GetRamInKB(&memTotal, &memFree);

	LOG4CPLUS_INFO_FMT(this->_logger,
			"NODE INFO - free RAM memory => %f MB, total RAM memory => %f MB",
			(float )this->memFree / 1024, (float )this->memTotal / 1024);
}

void NodeInfo::GetRamInKB(int* ramTotal, int* ramFree)
{
	FILE *meminfo = fopen("/proc/meminfo", "r");
//	if (meminfo == NULL)
//        ... // handle error


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

NodeInfo::NodeInfo()
{
	// TODO Auto-generated constructor stub

}

NodeInfo::~NodeInfo()
{
	// TODO Auto-generated destructor stub
}
}
}
