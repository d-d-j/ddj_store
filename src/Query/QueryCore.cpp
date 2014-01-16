#include "QueryCore.h"

namespace ddj {
namespace query {

	QueryCore::QueryCore(CudaController* cudaController) : _logger(Logger::getRoot()),_config(Config::GetInstance())
	{
		this->_cudaController = cudaController;
		this->_enableCompression = this->_config->GetIntValue("ENABLE_COMPRESSION");
	}

	QueryCore::~QueryCore(){}

	size_t QueryCore::ExecuteQuery(void** queryResult, Query* query, boost::container::vector<ullintPair>* dataLocationInfo)
	{
		// Read and copy data from mainMemoryPointer to temporary data buffer
		// Update dalaLocationInfo to contain info about location in tempDataBuffer
		void* tempDataBuffer = nullptr;
		size_t tempDataSize = this->mapData(&tempDataBuffer, dataLocationInfo);

		// Decompress temporary data buffer
		storeElement* decompressedBuffer = static_cast<storeElement*>(tempDataBuffer);
		size_t decompressedDataSize = tempDataSize;
		if(this->_enableCompression)
		{
			decompressedDataSize =
					this->decompressData(tempDataBuffer, tempDataSize, &decompressedBuffer, dataLocationInfo);
			cudaFree( tempDataBuffer );
		}

		// Filter to set of tags and time periods specified in query (only if set is not empty)
		size_t filteredDataSize = this->filterData(decompressedBuffer, decompressedDataSize, query, dataLocationInfo);

		// Aggregate all mapped and filtered data
		size_t aggregatedDataSize =
				this->aggregateData(decompressedBuffer, filteredDataSize, query, queryResult);

		cudaFree( decompressedBuffer );
		return aggregatedDataSize;
	}

	/***************************/
	/* DATA MANAGEMENT METHODS */
	/***************************/

	size_t QueryCore::aggregateData(storeElement* elements, size_t dataSize, Query* query, void** result)
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
		else
		{
			return this->_aggregationFunctions[query->aggregationType](elements, dataSize, result, query);
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
				size += dli.length();
			}
			CUDA_CHECK_RETURN( cudaMalloc(data, size) );

			// Copy fragments of mainGpuArray specified in dataLocationInfo vector to mapped data array
			// TODO: SPEED UP THIS COPING
			int position = 0;
			int oldPosition = 0;
			BOOST_FOREACH(ullintPair &dli, *dataLocationInfo)
			{
				ullint length = dli.length();
				char* source = (char*)(*data)+position;
				char* target = (char*)mainGpuArray+dli.first;
				CUDA_CHECK_RETURN(
						cudaMemcpy(	source,
									target,
									length * sizeof(char),
									cudaMemcpyDeviceToDevice)
				);
				oldPosition = position;
				position += length;
				// set this data location info to location in mapped data array
				dli.first = oldPosition;
				dli.second = position-1;
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

	size_t QueryCore::decompressData(void* data, size_t size, storeElement** elements,
			boost::container::vector<ullintPair>* dataLocationInfo)
	{
		compression::Compression c;
		size_t trunkSize = 0;
		size_t allDataSize = 0;
		boost::container::vector<std::pair<storeElement*, size_t> > decompressedTrunks;
		storeElement* trunk;
		BOOST_FOREACH(ullintPair &dli, *dataLocationInfo)
		{
			trunkSize = c.DecompressTrunk((char*)data+dli.first, dli.length(), &trunk);
			decompressedTrunks.push_back({trunk, trunkSize});
			allDataSize += trunkSize;
		}
		char* result;
		CUDA_CHECK_RETURN( cudaMalloc((void**)&result, allDataSize) );
		int position = 0;
		for(unsigned int i=0; i<decompressedTrunks.size(); i++)
		{
			trunk = decompressedTrunks[i].first;
			trunkSize = decompressedTrunks[i].second;
			CUDA_CHECK_RETURN( cudaMemcpy(result+position, trunk, trunkSize, cudaMemcpyDeviceToDevice) );
			position += trunkSize;
		}
		(*elements) = (storeElement*)result;
		return allDataSize;
	}

	size_t QueryCore::filterData(storeElement* elements, size_t dataSize, Query* query,
			boost::container::vector<ullintPair>* dataLocationInfo)
	{
		if(query && (query->tags.size() || query->timePeriods.size()))
		{
			if(query->aggregationType == AggregationType::Integral)
			{
				size_t size = gpu_filterData_in_trunks(
						elements,
						dataSize,
						query,
						dataLocationInfo->data(),
						dataLocationInfo->size());

				auto it = std::remove_if(
					dataLocationInfo->begin(),
					dataLocationInfo->end(),
					[](const ullintPair &pair){ return (pair.first - pair.second) == 1; }
				);
				dataLocationInfo->resize(it- dataLocationInfo->begin());

				query->aggregationData = dataLocationInfo;
				return size;
			}
			return gpu_filterData(elements, dataSize, query);
		}
		else return dataSize;
	}

} /* namespace query */
} /* namespace ddj */
