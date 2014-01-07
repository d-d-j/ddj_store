#include "QueryAggregation.h"

namespace ddj {
namespace query {

	QueryAggregation::QueryAggregation() {
		propagateAggregationMethods();
	}

	QueryAggregation::~QueryAggregation() {}

	/***********************/
	/* AGGREGATION MATHODS */
	/***********************/

	void QueryAggregation::propagateAggregationMethods()
	{
		// ADD
		this->_aggregationFunctions.insert({ AggregationType::Sum, boost::bind(&QueryAggregation::sum, this, _1, _2, _3, _4) });
		// MIN
		this->_aggregationFunctions.insert({ AggregationType::Min, boost::bind(&QueryAggregation::min, this, _1, _2, _3, _4) });
		// MAX
		this->_aggregationFunctions.insert({ AggregationType::Max, boost::bind(&QueryAggregation::max, this, _1, _2, _3, _4) });
		// AVERAGE
		this->_aggregationFunctions.insert({ AggregationType::Average, boost::bind(&QueryAggregation::average, this, _1, _2, _3, _4) });
		// STDDEVIATION
		this->_aggregationFunctions.insert({ AggregationType::StdDeviation, boost::bind(&QueryAggregation::variance, this, _1, _2, _3, _4) });
		// VARIANCE
		this->_aggregationFunctions.insert({ AggregationType::Variance, boost::bind(&QueryAggregation::variance, this, _1, _2, _3, _4) });
		// SKEWNESS
		this->_aggregationFunctions.insert({ AggregationType::Skewness, boost::bind(&QueryAggregation::skewness, this, _1, _2, _3, _4) });
		// KURTOSIS
		this->_aggregationFunctions.insert({ AggregationType::Kurtosis, boost::bind(&QueryAggregation::kurtosis, this, _1, _2, _3, _4) });
		// DIFFERENTIAL
		this->_aggregationFunctions.insert({ AggregationType::Differential, boost::bind(&QueryAggregation::differential, this, _1, _2, _3, _4) });
		// INTEGRAL
		this->_aggregationFunctions.insert({ AggregationType::Integral, boost::bind(&QueryAggregation::integral, this, _1, _2, _3, _4) });
		// HISTOGRAM ON VALUES
		this->_aggregationFunctions.insert({ AggregationType::Histogram_Value, boost::bind(&QueryAggregation::histogramValue, this, _1, _2, _3, _4) });
		// HISTOGRAM ON TIME
		this->_aggregationFunctions.insert({ AggregationType::Histogram_Time, boost::bind(&QueryAggregation::histogramTime, this, _1, _2, _3, _4) });
	}

	size_t QueryAggregation::sum(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_sum(elements, dataSize, result);
		else return 0;
	}

	size_t QueryAggregation::min(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_min(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::max(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_max(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::average(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_average(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::variance(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_variance(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::skewness(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_skewness(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::kurtosis(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_kurtosis(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::differential(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		return 0;
	}

	size_t QueryAggregation::integral(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize)
		{
			ullintPairVector* dataLocationInfo = static_cast<ullintPairVector*>(query->aggregationData);
			return gpu_trunk_integral(elements, dataSize, result, dataLocationInfo->data(), dataLocationInfo->size());
		}
		return 0;
	}

	size_t QueryAggregation::histogramValue(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize)
		{
			data::histogramValueData* data = static_cast<data::histogramValueData*>(query->aggregationData);

			//CREATE BUCKETS
			float2* buckets = new float2[data->bucketCount];
			float bucketSize = (data->max - data->min) / (float)data->bucketCount;
			float value = data->min;
			for(int i=0; i<data->bucketCount; i++)
			{
				buckets[i].x = value;
				value+=bucketSize;
				buckets[i].y = value;
			}

			//CALCULATE HISTOGRAM
			size_t size = gpu_histogram_value(elements, dataSize, result, buckets, data->bucketCount);

			//RELEASE BUCKETS
			delete [] buckets;

			return size;
		}
		return 0;
	}

	size_t QueryAggregation::histogramTime(storeElement* elements, size_t dataSize, void** result, Query* query)
	{
		(*result) = nullptr;
		if(dataSize)
		{
			data::histogramTimeData* data = static_cast<data::histogramTimeData*>(query->aggregationData);

			//CREATE BUCKETS
			ullint2* buckets = new ullint2[data->bucketCount];
			ullint bucketSize = (data->max - data->min) / (ullint)data->bucketCount;
			ullint value = data->min;
			for(int i=0; i<data->bucketCount; i++)
			{
				buckets[i].x = value;
				value+=bucketSize;
				buckets[i].y = value;
			}

			//CALCULATE HISTOGRAM
			size_t size = gpu_histogram_time(elements, dataSize, result, buckets, data->bucketCount);

			//RELEASE BUCKETS
			delete [] buckets;

			return size;
		}
		return 0;
	}

} /* namespace query */
} /* namespace ddj */
