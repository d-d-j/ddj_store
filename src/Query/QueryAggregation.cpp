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
		this->_aggregationFunctions.insert({ AggregationType::Sum, boost::bind(&QueryAggregation::sum, this, _1, _2, _3) });
		// MIN
		this->_aggregationFunctions.insert({ AggregationType::Min, boost::bind(&QueryAggregation::min, this, _1, _2, _3) });
		// MAX
		this->_aggregationFunctions.insert({ AggregationType::Max, boost::bind(&QueryAggregation::max, this, _1, _2, _3) });
		// AVERAGE
		this->_aggregationFunctions.insert({ AggregationType::Average, boost::bind(&QueryAggregation::average, this, _1, _2, _3) });
		// STDDEVIATION
		this->_aggregationFunctions.insert({ AggregationType::StdDeviation, boost::bind(&QueryAggregation::variance, this, _1, _2, _3) });
		// VARIANCE
		this->_aggregationFunctions.insert({ AggregationType::Variance, boost::bind(&QueryAggregation::variance, this, _1, _2, _3) });
		// DIFFERENTIAL
		this->_aggregationFunctions.insert({ AggregationType::Differential, boost::bind(&QueryAggregation::differential, this, _1, _2, _3) });
		// INTEGRAL
		this->_aggregationFunctions.insert({ AggregationType::Integral, boost::bind(&QueryAggregation::integral, this, _1, _2, _3) });
	}

	size_t QueryAggregation::sum(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_sum(elements, dataSize, result);
		else return 0;
	}

	size_t QueryAggregation::min(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_min(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::max(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_max(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::average(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_average(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::variance(storeElement* elements, size_t dataSize, void** result)
	{
		(*result) = nullptr;
		if(dataSize) return gpu_variance(elements, dataSize, result);
		return 0;
	}

	size_t QueryAggregation::differential(storeElement* elements, size_t dataSize, void** result)
	{
		return 0;
	}

	size_t QueryAggregation::integral(storeElement* elements, size_t dataSize, void** result)
	{
		return 0;
	}

} /* namespace query */
} /* namespace ddj */
