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
        Logger _logger;
		Config* _config;
		store::CudaCommons _cudaCommons;


	public:
		StoreInfoCore() : _logger(Logger::getRoot()), _config(Config::GetInstance()) {};

		void GetRamInKB(int* ramTotal, int* ramFree);

		size_t GetNodeInfo(void** result);
	};

}
}
#endif /* STOREINFOCORE_H_ */
