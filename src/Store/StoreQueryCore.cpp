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
		propagateAggregationMethods();
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

	void StoreQueryCore::propagateAggregationMethods()
	{
		// ADD
		this->_aggregationFunctions.insert({ AggregationType::Add, boost::bind(&StoreQueryCore::add, this, _1, _2, _3) });
		// MIN
		this->_aggregationFunctions.insert({ AggregationType::Min, boost::bind(&StoreQueryCore::min, this, _1, _2, _3) });
		// MAX
		this->_aggregationFunctions.insert({ AggregationType::Max, boost::bind(&StoreQueryCore::max, this, _1, _2, _3) });
		// AVERAGE
		this->_aggregationFunctions.insert({ AggregationType::Average, boost::bind(&StoreQueryCore::average, this, _1, _2, _3) });
		// STDDEVIATION
		this->_aggregationFunctions.insert({ AggregationType::StdDeviation, boost::bind(&StoreQueryCore::stdDeviation, this, _1, _2, _3) });
		// COUNT
		this->_aggregationFunctions.insert({ AggregationType::Count, boost::bind(&StoreQueryCore::count, this, _1, _2, _3) });
		// VARIANCE
		this->_aggregationFunctions.insert({ AggregationType::Variance, boost::bind(&StoreQueryCore::variance, this, _1, _2, _3) });
		// DIFFERENTIAL
		this->_aggregationFunctions.insert({ AggregationType::Differential, boost::bind(&StoreQueryCore::differential, this, _1, _2, _3) });
		// INTEGRAL
		this->_aggregationFunctions.insert({ AggregationType::Integral, boost::bind(&StoreQueryCore::integral, this, _1, _2, _3) });
	}

	size_t StoreQueryCore::add(storeElement* elements, size_t dataSize, storeElement** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_add_values(elements, dataSize, result);
		else return 0;
	}

	size_t StoreQueryCore::min(storeElement* elements, size_t dataSize, storeElement** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_min_from_values(elements, dataSize, result);
		return 0;
	}

	size_t StoreQueryCore::max(storeElement* elements, size_t dataSize, storeElement** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_max_from_values(elements, dataSize, result);
		return 0;
	}

	size_t StoreQueryCore::average(storeElement* elements, size_t dataSize, storeElement** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_average_from_values(elements, dataSize, result);
		return 0;
	}

	size_t StoreQueryCore::stdDeviation(storeElement* elements, size_t dataSize, storeElement** result)
	{
		return 0;
	}

	size_t StoreQueryCore::count(storeElement* elements, size_t dataSize, storeElement** result)
	{
		return 0;
	}

	size_t StoreQueryCore::variance(storeElement* elements, size_t dataSize, storeElement** result)
	{
		return 0;
	}

	size_t StoreQueryCore::differential(storeElement* elements, size_t dataSize, storeElement** result)
	{
		return 0;
	}

	size_t StoreQueryCore::integral(storeElement* elements, size_t dataSize, storeElement** result)
	{
		return 0;
	}

} /* namespace store */
} /* namespace ddj */
