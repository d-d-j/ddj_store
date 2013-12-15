/*
 * NodeInfo.h
 *
 *  Created on: Dec 15, 2013
 *      Author: dud
 */

#ifndef NODEINFO_H_
#define NODEINFO_H_

#include "../Cuda/CudaCommons.h"
#include "../Core/Logger.h"
#include "../Core/Config.h"

namespace ddj{
namespace store{
	class NodeInfo
	{
		Config* _config = Config::GetInstance();
		Logger _logger = Logger::getRoot();
		store::CudaCommons _cudaCommons;

		int memTotal;
		int memFree;
		int cpuPercent;
		size_t gpuMemTotal;
		size_t gpuMemFree;


	public:
		NodeInfo();
		virtual ~NodeInfo();
		void GetRamInKB(int* ramTotal, int* ramFree);

		void FillNodeInfo();
	};

}
}
#endif /* NODEINFO_H_ */
