/*
 * TaskReducer.cpp
 *
 *  Created on: 09-01-2014
 *      Author: ghash
 */

#include "TaskReducer.h"

namespace ddj {
namespace task {

	size_t TaskReducer::Reduce(query::Query* query, void* taskResult, size_t resultSize, void** reducedResult)
	{
		if(query == nullptr || resultSize == 0)
		{
			(*reducedResult) = nullptr;
			return 0;
		}
		switch(query->aggregationType)
		{
			case AggregationType::Histogram_Value:
				return reduceHistogramValue(
						static_cast<query::data::histogramValueData*>(query->aggregationData),
						taskResult,
						resultSize,
						reducedResult);
				break;
			case AggregationType::Histogram_Time:
				return reduceHistogramTime(
						static_cast<query::data::histogramTimeData*>(query->aggregationData),
						taskResult,
						resultSize,
						reducedResult);
				break;
			default:
				(*reducedResult) = nullptr;
				return 0;
				break;
		}
	}

	size_t TaskReducer::reduceHistogramValue(
			query::data::histogramValueData* data,
			void* taskResult,
			size_t resultSize,
			void** reducedResult)
	{
		int* unaggregatedData = static_cast<int*>(taskResult);
		int* result = new int[data->bucketCount];
		int resultCount = resultSize/sizeof(int)/data->bucketCount;
		memset(result, 0, sizeof(int)*data->bucketCount);
		for(int i=0; i<resultCount; i++)
		{
			for(int j=0; j<data->bucketCount; j++)
			{
				int idx = i*data->bucketCount + j;
				result[j] += unaggregatedData[idx];
			}
		}
		(*reducedResult) = result;
		return sizeof(int)*data->bucketCount;
	}

	size_t TaskReducer::reduceHistogramTime(
			query::data::histogramTimeData* data,
			void* taskResult,
			size_t resultSize,
			void** reducedResult)
	{
		int* unaggregatedData = static_cast<int*>(taskResult);
		int resultCount = resultSize/sizeof(int)/data->bucketCount;
		int* result = new int[data->bucketCount];
		memset(result, 0, sizeof(int)*data->bucketCount);
		for(int i=0; i<resultCount; i++)
		{
			for(int j=0; j<data->bucketCount; j++)
			{
				int idx = i*data->bucketCount + j;
				result[j] += unaggregatedData[idx];
			}
		}
		(*reducedResult) = result;
		return sizeof(int)*data->bucketCount;
	}

} /* namespace task */
} /* namespace ddj */
