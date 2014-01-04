#include "QueryCore.h"

namespace ddj {
namespace query {

	QueryCore::QueryCore(CudaController* cudaController)
	{
		this->_cudaController = cudaController;
		propagateAggregationMethods();
	}

	QueryCore::~QueryCore(){}

	size_t QueryCore::ExecuteQuery(void** queryResult, Query* query, boost::container::vector<ullintPair>* dataLocationInfo)
	{
		// Read and copy data from mainMemoryPointer to temporary data buffer
		// Update dalaLocationInfo to contain info about location in tempDataBuffer
		storeElement* tempDataBuffer = nullptr;
		size_t tempDataSize = this->mapData((void**)&tempDataBuffer, dataLocationInfo);

		// TODO: Decompress temporary data buffer

		// Filter to set of tags and time periods specified in query (only if set is not empty)
		size_t filteredDataSize = this->filterData(tempDataBuffer, tempDataSize, query);

		// Aggregate all mapped and filtered data
		void* aggregatedData;
		size_t aggregatedDataSize =
				this->aggregateData(tempDataBuffer, filteredDataSize, query, &aggregatedData);

		// Return results
		(*queryResult) = aggregatedData;
		return aggregatedDataSize;
	}

	/***************************/
	/* DATA MANAGEMENT METHODS */
	/***************************/

	size_t QueryCore::aggregateData(storeElement* elements, size_t dataSize, Query* query, void** result,
			boost::container::vector<ullintPair>* dataLocationInfo)
	{
		if (dataSize == 0)
		{
			(*result) = nullptr;
			return 0;
		}
		else if(query->aggregationType == AggregationType::None)
		{
			(*result) = malloc(dataSize);
			CUDA_CHECK_RETURN( cudaMemcpy((*result), elements, dataSize, cudaMemcpyDeviceToHost) );
			return dataSize;
		}
		else if(query->aggregationType == AggregationType::Integral)
		{
			return gpu_trunk_integral(elements, dataSize, result,
					dataLocationInfo->data(), dataLocationInfo->size());
		}
		else
		{
			return this->_aggregationFunctions[query->aggregationType](elements, dataSize, result);
		}
	}

	size_t QueryCore::mapData(void** data, boost::container::vector<ullintPair>* dataLocationInfo)
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
			int oldPosition = 0;
			BOOST_FOREACH(ullintPair &dli, *dataLocationInfo)
			{
				CUDA_CHECK_RETURN(
						cudaMemcpy((char*)(*data)+position,
						(char*)mainGpuArray+dli.first,
						(dli.second-dli.first+1),
						cudaMemcpyDeviceToDevice));
				oldPosition = position;
				position += (dli.second-dli.first+1);
				// set this data location info to location in mapped data array
				dli.first = oldPosition;
				dli.second = position;
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

	storeElement* QueryCore::decompressData(void* data, size_t* size)
	{
		return nullptr;
	}

	size_t QueryCore::filterData(storeElement* elements, size_t dataSize, Query* query,
			boost::container::vector<ullintPair>* dataLocationInfo)
	{
		if(query && (query->tags.size() || query->timePeriods.size()))
		{
			if(query->aggregationType == AggregationType::Integral)
			{
				return gpu_filterData_in_trunks(elements, dataSize, query, dataLocationInfo->data(), dataLocationInfo->size());
			}
			return gpu_filterData(elements, dataSize, query);
		}
		else return dataSize;
	}

	/***********************/
	/* AGGREGATION MATHODS */
	/***********************/

	void QueryCore::propagateAggregationMethods()
	{
		// ADD
		this->_aggregationFunctions.insert({ AggregationType::Sum, boost::bind(&QueryCore::sum, this, _1, _2, _3) });
		// MIN
		this->_aggregationFunctions.insert({ AggregationType::Min, boost::bind(&QueryCore::min, this, _1, _2, _3) });
		// MAX
		this->_aggregationFunctions.insert({ AggregationType::Max, boost::bind(&QueryCore::max, this, _1, _2, _3) });
		// AVERAGE
		this->_aggregationFunctions.insert({ AggregationType::Average, boost::bind(&QueryCore::average, this, _1, _2, _3) });
		// STDDEVIATION
		this->_aggregationFunctions.insert({ AggregationType::StdDeviation, boost::bind(&QueryCore::variance, this, _1, _2, _3) });
		// VARIANCE
		this->_aggregationFunctions.insert({ AggregationType::Variance, boost::bind(&QueryCore::variance, this, _1, _2, _3) });
		// DIFFERENTIAL
		this->_aggregationFunctions.insert({ AggregationType::Differential, boost::bind(&QueryCore::differential, this, _1, _2, _3) });
		// INTEGRAL
		this->_aggregationFunctions.insert({ AggregationType::Integral, boost::bind(&QueryCore::integral, this, _1, _2, _3) });
	}

	size_t QueryCore::sum(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_sum(elements, dataSize, result);
		else return 0;
	}

	size_t QueryCore::min(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_min(elements, dataSize, result);
		return 0;
	}

	size_t QueryCore::max(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_max(elements, dataSize, result);
		return 0;
	}

	size_t QueryCore::average(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_average(elements, dataSize, result);
		return 0;
	}

	size_t QueryCore::variance(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_variance(elements, dataSize, result);
		return 0;
	}

	size_t QueryCore::differential(storeElement* elements, size_t dataSize, void** result)
	{
		return 0;
	}

	size_t QueryCore::integral(storeElement* elements, size_t dataSize, void** result)
	{
		return 0;
	}

} /* namespace query */
} /* namespace ddj */
