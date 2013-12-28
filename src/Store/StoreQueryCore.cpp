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

	size_t StoreQueryCore::ExecuteQuery(void** queryResult, storeQuery* query, boost::container::vector<ullintPair>* dataLocationInfo)
	{
		// Read and copy data from mainMemoryPointer to temporary data buffer
		void* tempDataBuffer = nullptr;
		size_t tempDataSize = this->mapData(&tempDataBuffer, dataLocationInfo);
		LOG4CPLUS_DEBUG(this->_logger, "StoreQueryCore - tempDataElemCount = " << tempDataSize/sizeof(storeElement));
		// TODO: Decompress temporary data buffer

		// Filter to set of tags specified in query (only if set is not empty)
		size_t filteredStoreSize = this->filterData((storeElement*)tempDataBuffer, tempDataSize, query);
		LOG4CPLUS_DEBUG(this->_logger, "StoreQueryCore - filteredDataElemCount = " << filteredStoreSize/sizeof(storeElement));

		// TODO: Aggregate all mapped data

		// Set queryResult, clean and return result size
		(*queryResult) = malloc(filteredStoreSize);
		CUDA_CHECK_RETURN( cudaMemcpy((*queryResult), tempDataBuffer, filteredStoreSize, cudaMemcpyDeviceToHost) );
		CUDA_CHECK_RETURN( cudaFree(tempDataBuffer) );
		return filteredStoreSize;
	}

	/***************************/
	/* DATA MANAGEMENT METHODS */
	/***************************/

	size_t StoreQueryCore::mapData(void** data, boost::container::vector<ullintPair>* dataLocationInfo)
	{
		void* mainGpuArray = _cudaController->GetMainMemoryPointer();
		size_t size = 0;
		if(dataLocationInfo && dataLocationInfo->size())	// select mapped data
		{
			// Calculate mapped data size and create new device array for mapped data
			BOOST_FOREACH(ullintPair &dli, *dataLocationInfo)
			{
				size += (dli.second-dli.first+1);
			}
			CUDA_CHECK_RETURN( cudaMalloc(data, size) );

			// Copy fragments of mainGpuArray specified in dataLocationInfo vector to mapped data array
			// TODO: SPEED UP THIS COPING
			int position = 0;
			BOOST_FOREACH(ullintPair &dli, *dataLocationInfo)
			{
				CUDA_CHECK_RETURN(
						cudaMemcpy((char*)(*data)+position,
						(char*)mainGpuArray+dli.first,
						(dli.second-dli.first+1),
						cudaMemcpyDeviceToDevice));
				position += (dli.second-dli.first+1);
			}
		}
		else if(dataLocationInfo == nullptr)
		{
			// Get mainGpuArray data size (offset)
			size = _cudaController->GetMainMemoryOffset();
			CUDA_CHECK_RETURN( cudaMalloc(data, size) );

			// Copy whole data from mainGpuArray to new device array (only part containing some data)
			CUDA_CHECK_RETURN( cudaMemcpy(*data, mainGpuArray, size, cudaMemcpyDeviceToDevice) );
		}
		return size;
	}

	storeElement* StoreQueryCore::decompressData(void* data, size_t* size)
	{
		return nullptr;
	}

	size_t StoreQueryCore::filterData(storeElement* elements, size_t dataSize, storeQuery* query)
	{
		if(query && query->tags.size())
			return gpu_filterData(elements, dataSize, query);
		else return dataSize;
	}

	/***********************/
	/* AGGREGATION MATHODS */
	/***********************/

	float add(storeElement* elements, int count)
	{
		return 0.0f;
	}

	float average(storeElement* elements, int count)
	{
		return 0.0f;
	}

	storeElement* max(storeElement* elements, int count)
	{
		return nullptr;
	}

	storeElement* min(storeElement* elements, int count)
	{
		return nullptr;
	}

} /* namespace store */
} /* namespace ddj */
