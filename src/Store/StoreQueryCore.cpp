/*
 * QueryCore.cpp
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#include "StoreQueryCore.h"

namespace ddj {
namespace store {

	StoreQueryCore::StoreQueryCore(CudaController* cudaController)
	{
		this->_cudaController = cudaController;
	}

	StoreQueryCore::~StoreQueryCore(){}

	/*
	size_t StoreQueryCore::SelectAll(void** queryResult)
	{
		void* mainMemoryPointer = this->_cudaController->GetMainMemoryPointer();
		ullint offset = this->_cudaController->GetMainMemoryOffset();
		CUDA_CHECK_RETURN( cudaMallocHost(queryResult, offset) );
		CUDA_CHECK_RETURN( cudaMemcpy(*queryResult, mainMemoryPointer, offset, cudaMemcpyDeviceToHost) );
		return offset;
	}
	*/

	size_t StoreQueryCore::ExecuteQuery(void** queryResult, storeQuery* query, boost::container::vector<ullintPair>* dataLocationInfo)
	{
		// Read and copy data from mainMemoryPointer to temporary data buffer

		// Decompress temporary data buffer

		// Filter to set of tags specified in query (only if set is not empty)

		// Aggregate all mapped data

		// Set queryResult, clean and return result size
		return 0;
	}

	/* DATA MANAGEMENT METHODS */


	size_t StoreQueryCore::mapData(void** data, boost::container::vector<ullintPair>* dataLocationInfo)
	{
		return 0;
	}

	storeElement* StoreQueryCore::decompressData(void* data, size_t size)
	{
		return nullptr;
	}

	size_t StoreQueryCore::filterData(storeElement* elements, storeQuery* query)
	{
		return 0;
	}

	/* AGGREGATION MATHODS */
	void StoreQueryCore::add(storeQuery* query)
	{

	}

} /* namespace store */
} /* namespace ddj */
