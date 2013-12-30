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

		// TODO: Decompress temporary data buffer

		// Filter to set of tags and time periods specified in query (only if set is not empty)
		size_t filteredDataSize = this->filterData((storeElement*)tempDataBuffer, tempDataSize, query);

		// Aggregate all mapped and filtered data
		size_t aggregatedDataSize = this->aggregateData((storeElement**)&tempDataBuffer, filteredDataSize, query);

		// Set queryResult, clean and return result size
		(*queryResult) = nullptr;
		if(aggregatedDataSize)
		{
			(*queryResult) = malloc(aggregatedDataSize);
			CUDA_CHECK_RETURN( cudaMemcpy((*queryResult), tempDataBuffer, aggregatedDataSize, cudaMemcpyDeviceToHost) );
		}
		return aggregatedDataSize;
	}

	/***************************/
	/* DATA MANAGEMENT METHODS */
	/***************************/

	size_t StoreQueryCore::aggregateData(storeElement** elements, size_t dataSize, storeQuery* query)
	{
		if (dataSize == 0)
		{
			CUDA_CHECK_RETURN( cudaFree(*elements) );
			(*elements) = nullptr;
			return 0;
		}
		if(query->aggregationType != AggregationType::None)
		{
			storeElement* aggregatedData = nullptr;
			size_t aggregatedDataSize =
					this->_aggregationFunctions[query->aggregationType](*elements, dataSize, &aggregatedData);
			CUDA_CHECK_RETURN( cudaFree(*elements) );
			(*elements) = aggregatedData;
			return aggregatedDataSize;
		}
		return dataSize;
	}

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
		if(query && (query->tags.size() || query->timePeriods.size()))
		{
			return gpu_filterData(elements, dataSize, query);
		}
		else return dataSize;
	}

	/***********************/
	/* AGGREGATION MATHODS */
	/***********************/

	size_t StoreQueryCore::add(storeElement* elements, int count, storeElement** result)
	{
		(*result) = nullptr;
		return 0;
	}

	size_t StoreQueryCore::average(storeElement* elements, int count, storeElement** result)
	{
		(*result) = nullptr;
		return 0;
	}

	size_t StoreQueryCore::max(storeElement* elements, int count, storeElement** result)
	{
		(*result) = nullptr;
		return 0;
	}

	size_t StoreQueryCore::min(storeElement* elements, int count, storeElement** result)
	{
		(*result) = nullptr;
		return 0;
	}

} /* namespace store */
} /* namespace ddj */
