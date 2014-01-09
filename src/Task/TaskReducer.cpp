/*
 * TaskReducer.cpp
 *
 *  Created on: 09-01-2014
 *      Author: ghash
 */

#include "TaskReducer.h"

namespace ddj {
namespace task {

	void TaskReducer::Reduce(query::Query* query, void* taskResult, int resultCount, void** reducedResult)
	{
		switch(query->aggregationType)
		{
			case AggregationType::Histogram_Value:
				reduceHistogramValue(
						static_cast<query::data::histogramValueData*>(query->aggregationData),
						taskResult,
						resultCount,
						reducedResult);
				break;
			case AggregationType::Histogram_Time:
				reduceHistogramTime(
						static_cast<query::data::histogramTimeData*>(query->aggregationData),
						taskResult,
						resultCount,
						reducedResult);
				break;
			default:
				(*reducedResult) = nullptr;
				break;
		}
	}

	void TaskReducer::reduceHistogramValue(
			query::data::histogramValueData* data,
			void* taskResult,
			int resultCount,
			void** reducedResult)
	{
		int* unaggregatedData = static_cast<int*>(taskResult);
		int* result = new int[data->bucketCount];
		memset(result, 0, sizeof(int)*data->bucketCount);
		for(int i=0; i<resultCount; i++)
			for(int j=0; j<data->bucketCount; j++)
				result[j] += unaggregatedData[i*data->bucketCount + j];
		(*reducedResult) = result;
	}

	void TaskReducer::reduceHistogramTime(
			query::data::histogramTimeData* data,
			void* taskResult,
			int resultCount,
			void** reducedResult)
	{
		int* unaggregatedData = static_cast<int*>(taskResult);
		int* result = new int[data->bucketCount];
		memset(result, 0, sizeof(int)*data->bucketCount);
		for(int i=0; i<resultCount; i++)
			for(int j=0; j<data->bucketCount; j++)
				result[j] += unaggregatedData[i*data->bucketCount + j];
		(*reducedResult) = result;
	}

} /* namespace task */
} /* namespace ddj */
