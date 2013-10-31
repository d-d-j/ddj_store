/*
 * QueryCore.cpp
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#include "QueryCore.h"

namespace ddj {
namespace store {

QueryCore::QueryCore(CudaController* cudaController)
{
	this->_cudaController = cudaController;
}

QueryCore::~QueryCore(){}

void* QueryCore::GetAllData(size_t &size)
{
	if(size==0) return NULL;
	void* mainMemoryPointer = this->_cudaController->GetMainMemoryPointer();
	ullint offset = this->_cudaController->GetMainMemoryOffset();
	void* hostData;
	CUDA_CHECK_RETURN( cudaMallocHost(&hostData,offset) );
	CUDA_CHECK_RETURN( cudaMemcpy(hostData, mainMemoryPointer, offset, cudaMemcpyDeviceToHost) );
	size = offset;
	return hostData;
}

} /* namespace store */
} /* namespace ddj */
