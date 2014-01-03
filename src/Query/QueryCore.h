#ifndef QUERYCORE_H_
#define QUERYCORE_H_

#include "Query.h"
#include "../Store/StoreElement.h"
#include "../Core/Logger.h"
#include "../Cuda/CudaController.h"
#include "../Cuda/CudaIncludes.h"
#include "../Cuda/CudaQuery.cuh"
#include <gtest/gtest.h>
#include <boost/foreach.hpp>
#include <boost/unordered_map.hpp>

namespace ddj {
namespace query {

using namespace store;

	class QueryCore : public boost::noncopyable
	{
		typedef boost::function<size_t (storeElement* elements, size_t size, storeElement** result)> aggregationFunc;

	private:
		CudaController* _cudaController;
		Logger _logger = Logger::getRoot();
		boost::unordered_map<int, aggregationFunc> _aggregationFunctions;

	public:
		QueryCore(CudaController* cudaController);
		virtual ~QueryCore();

		/*
		 * Description:
		 * Method executing query with optional selected dataLocations;
		 * If no dataLocation provided query is executed to all data in store
		 * Returns:
		 * 	size of data produced by query
		 * Output:
		 * 	result of query is returned to queryResult parameter
		 */
		size_t ExecuteQuery(void** queryResult, Query* query, boost::container::vector<ullintPair>* dataLocationInfo = nullptr);

	private:
		/***************************/
		/* DATA MANAGEMENT METHODS */
		/***************************/

		/*
		 * Description:
		 * Method aggregating dataSize data from elements using aggregation defined in query;
		 * If aggregation type isn't None, method releases elements and sets it to aggregation result;
		 * If dataSize equals 0 method should set elements to null and returns 0;
		 * Returns:
		 *  returns size of new elements data
		 * Output:
		 *  aggregated data is returned as elements array (old one is released)
		 */
		size_t aggregateData(storeElement** elements, size_t dataSize, Query* query);

		/*
		 * Description:
		 * Method mapping all data stored in GPU to selected data locations;
		 * Selects only parts of mainGpuArray specified by ullintPair, for example:
		 * if dataLocationInfo contains ullintPair (100,300) then a part or mainGpuArray
		 * from index 100 to index 300 (including 300) will be returned as new gpu (device) array in data parameter.
		 * Returns:
		 *  returns size of mapped data
		 * Output:
		 *  mapped data is returned as device array in data parameter
		 */
		size_t mapData(void** data, boost::container::vector<ullintPair>* dataLocationInfo = nullptr);

		/*
		 * Description:
		 * Method filtering an array of storeElements on GPU by tags
		 * It does nothing if no tags are specified in query;
		 * Otherwise, it moves all elements with tag equal to any of provided in query
		 * to the front of elements array, and returns a number of these elements.
		 * Returns:
		 *  number of elements moved to front (number of elements with specified tags)
		 * Output:
		 *  changed elements array
		 */
		size_t filterData(storeElement* elements, size_t dataSize, Query* query);

		storeElement* decompressData(void* data, size_t* size);

		/***********************/
		/* AGGREGATION MATHODS */
		/***********************/

		void propagateAggregationMethods();

		size_t add(storeElement* elements, size_t dataSize, storeElement** result);
		size_t min(storeElement* elements, size_t dataSize, storeElement** result);
		size_t max(storeElement* elements, size_t dataSize, storeElement** result);
		size_t average(storeElement* elements, size_t dataSize, storeElement** result);
		size_t stdDeviation(storeElement* elements, size_t dataSize, storeElement** result);
		size_t count(storeElement* elements, size_t dataSize, storeElement** result);
		size_t variance(storeElement* elements, size_t dataSize, storeElement** result);
		size_t differential(storeElement* elements, size_t dataSize, storeElement** result);
		size_t integral(storeElement* elements, size_t dataSize, storeElement** result);

	private:
		friend class QueryCoreTest;

		/*********/
		/* TESTS */
		/*********/

	//mapData
		FRIEND_TEST(QueryCoreTest, mapData_AllData);
		FRIEND_TEST(QueryCoreTest, mapData_ChooseOneTrunk);
		FRIEND_TEST(QueryCoreTest, mapData_ChooseManyTrunks);
	//filterData
		FRIEND_TEST(QueryCoreTest, filterData_AllData);
		FRIEND_TEST(QueryCoreTest, filterData_ExistingTags);
		FRIEND_TEST(QueryCoreTest, filterData_NonExistingTags);
		FRIEND_TEST(QueryCoreTest, filterData_ExistingTags_FromTimePeriod);
	//selectData
		FRIEND_TEST(QueryCoreTest, ExecuteQuery_SpecificTimeFrame_AllTags_NoAggregation);
		FRIEND_TEST(QueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_NoAggregation);
		FRIEND_TEST(QueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_SumAggregation);
	//add
		FRIEND_TEST(QueryCoreTest, add_Empty);
		FRIEND_TEST(QueryCoreTest, add_EvenNumberOfValues);
		FRIEND_TEST(QueryCoreTest, add_OddNumberOfValues);
	//min
		FRIEND_TEST(QueryCoreTest, min_Empty);
		FRIEND_TEST(QueryCoreTest, min_Positive);
		FRIEND_TEST(QueryCoreTest, min_Negative);
	//max
		FRIEND_TEST(QueryCoreTest, max_Empty);
		FRIEND_TEST(QueryCoreTest, max_Positive);
		FRIEND_TEST(QueryCoreTest, max_Negative);
	//average
		FRIEND_TEST(QueryCoreTest, average_Empty);
		FRIEND_TEST(QueryCoreTest, average_Linear);
		FRIEND_TEST(QueryCoreTest, average_Sinusoidal);
	//stdDeviation
		FRIEND_TEST(QueryCoreTest, stdDeviation_Empty);
		FRIEND_TEST(QueryCoreTest, stdDeviation_Simple);
		FRIEND_TEST(QueryCoreTest, stdDeviation_Linear);
	//count
		FRIEND_TEST(QueryCoreTest, count_Empty);
		FRIEND_TEST(QueryCoreTest, count_NonEmpty);
	//variance
		FRIEND_TEST(QueryCoreTest, variance_Empty);
		FRIEND_TEST(QueryCoreTest, variance_Simple);
		FRIEND_TEST(QueryCoreTest, variance_Linear);
	//differential
	//integral

	};

} /* namespace query */
} /* namespace ddj */
#endif /* QUERYCORE_H_ */
