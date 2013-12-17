#ifndef STOREINFOCORE_H_
#define STOREINFOCORE_H_

#include "../Cuda/CudaCommons.h"
#include "../Core/Logger.h"
#include "../Core/Config.h"

#include "StoreNodeInfo.h"

namespace ddj{
namespace store{
	class StoreInfoCore
	{
		Config* _config = Config::GetInstance();
		Logger _logger = Logger::getRoot();
		store::CudaCommons _cudaCommons;


	public:
		StoreInfoCore();

		void GetRamInKB(int* ramTotal, int* ramFree);

		size_t GetNodeInfo(void** result);
	};

}
}
#endif /* STOREINFOCORE_H_ */
