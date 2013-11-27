/*
 * CudaCommon.h
 *
 *  Created on: 19-11-2013
 *      Author: ghash
 */

#ifndef CUDACOMMON_H_
#define CUDACOMMON_H_

#include "CudaIncludes.h"
#include "../Core/Logger.h"
#include "../Core/Config.h"

namespace ddj {
namespace store {

class CudaCommons {
private:
	/* LOGGER & CONFIG */
	Logger _logger = Logger::getRoot();
	Config* _config = Config::GetInstance();

public:
	CudaCommons();
	virtual ~CudaCommons();

	/* CUDA DEVICES */
	int CudaGetDevicesCount();
	bool CudaCheckDeviceForRequirements(int n);
	int CudaGetDevicesCountAndPrint();

	/* CUDA MALLOC */
	cudaError_t CudaAllocateArray(size_t size, void** array);
	void CudaFreeMemory(void* devPtr);
};

} /* namespace store */
} /* namespace ddj */
#endif /* CUDACOMMON_H_ */
