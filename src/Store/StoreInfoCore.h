#ifndef STOREINFOCORE_H_
#define STOREINFOCORE_H_

#include "../Cuda/CudaCommons.h"
#include "../Core/Logger.h"
#include "../Core/Config.h"

namespace ddj{
namespace store{
	class StoreInfoCore
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
		StoreInfoCore();
		virtual ~StoreInfoCore();
		void GetRamInKB(int* ramTotal, int* ramFree);

		void FillNodeInfo();
	};

}
}
#endif /* STOREINFOCORE_H_ */
