#ifndef QUERYCORE_H_
#define QUERYCORE_H_

#include "Query.h"
#include "QueryAggregation.h"
#include "../Store/StoreElement.cuh"
#include "../Core/Logger.h"
#include "../Cuda/CudaController.h"
#include "../Cuda/CudaIncludes.h"
#include "../Cuda/CudaQuery.cuh"
#include <boost/foreach.hpp>


namespace ddj {
namespace query {

	using namespace store;

	class QueryCore : public boost::noncopyable, public QueryAggregation
	{
	private:
		CudaController* _cudaController;
		Logger _logger = Logger::getRoot();

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
		 * Method aggregates store elements (on device) of dataSize size using aggregation defined in query's aggregation
		 * type and return aggregated value to result on host;
		 * Store Elements should be already on device;
		 * If aggregation type is None, method copies elements from device to aggregation result on host;
		 * If dataSize equals 0 method should set aggregation result on host to null and returns 0;
		 * Returns:
		 *  returns size of new elements data
		 * Output:
		 *  aggregated data is returned as elements array (old one is released)
		 */
		size_t aggregateData(storeElement* elements, size_t dataSize, Query* query, void** result);

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
		size_t filterData(storeElement* elements, size_t dataSize, Query* query,
				boost::container::vector<ullintPair>* dataLocationInfo = nullptr);

		storeElement* decompressData(void* data, size_t* size);

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
		FRIEND_TEST(QueryCoreTest, filterData_inTrunks_Empty_Trunk);
	//mapAndFilterData
		FRIEND_TEST(QueryCoreTest, mapData_and_filterData_InTrunks_WithExistingTags_FromTimePeriod);
	//selectData
		FRIEND_TEST(QueryCoreTest, ExecuteQuery_SpecificTimeFrame_AllTags_NoAggregation);
		FRIEND_TEST(QueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_NoAggregation);
		FRIEND_TEST(QueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_SumAggregation);
	};

} /* namespace query */
} /* namespace ddj */
#endif /* QUERYCORE_H_ */
