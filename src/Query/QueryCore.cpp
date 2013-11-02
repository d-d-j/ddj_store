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

size_t QueryCore::SelectAll(void** queryResult)
{
	void* mainMemoryPointer = this->_cudaController->GetMainMemoryPointer();
	ullint offset = this->_cudaController->GetMainMemoryOffset();
	CUDA_CHECK_RETURN( cudaMallocHost(queryResult, offset) );
	CUDA_CHECK_RETURN( cudaMemcpy(*queryResult, mainMemoryPointer, offset, cudaMemcpyDeviceToHost) );
	return offset;
}

} /* namespace store */
} /* namespace ddj */
