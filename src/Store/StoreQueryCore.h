/*
 * QueryCore.h
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#ifndef QUERYCORE_H_
#define QUERYCORE_H_

#include "StoreQuery.h"
#include "StoreElement.h"
#include "../Core/Logger.h"
#include "../Cuda/CudaController.h"
#include "../Cuda/CudaIncludes.h"
#include "../Cuda/CudaQuery.cuh"
#include <gtest/gtest.h>
#include <boost/foreach.hpp>

namespace ddj {
namespace store {

	class StoreQueryCore : public boost::noncopyable
	{
	private:
		CudaController* _cudaController;
		Logger _logger = Logger::getRoot();

	public:
		StoreQueryCore(CudaController* cudaController);
		virtual ~StoreQueryCore();

		/*
		 * Description:
		 * Method executing query with optional selected dataLocations;
		 * If no dataLocation provided query is executed to all data in store
		 * Returns:
		 * 	size of data produced by query
		 * Output:
		 * 	result of query is returned to queryResult parameter
		 */
		size_t ExecuteQuery(void** queryResult, storeQuery* query, boost::container::vector<ullintPair>* dataLocationInfo = nullptr);

	private:
		/***************************/
		/* DATA MANAGEMENT METHODS */
		/***************************/

		/*
		 * Description:
		 * Method mapping all data stored in GPU to selected data locations;
		 * Selects only parts of mainGpuArray specified by ullintPair, for example:
		 * if dataLocationInfo contains ullintPair (100,300) then a part or mainGpuArray
		 * from index 100 to index 300 (including 300) will be returned as new gpu (device) array in data parameter
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
		 * to the front of elements array, and returns a number of these elements
		 * Returns:
		 *  number of elements moved to front (number of elements with specified tags)
		 * Output:
		 *  changed elements array
		 */
		size_t filterData(storeElement* elements, size_t dataSize, storeQuery* query);

		storeElement* decompressData(void* data, size_t* size);

		/***********************/
		/* AGGREGATION MATHODS */
		/***********************/

		float add(storeElement* elements, int count);
		float average(storeElement* elements, int count);
		storeElement* max(storeElement* elements, int count);
		storeElement* min(storeElement* elements, int count);

	private:
		friend class StoreQueryCoreTest;

		/*********/
		/* TESTS */
		/*********/

	//mapData
		FRIEND_TEST(StoreQueryCoreTest, mapData_AllData);
		FRIEND_TEST(StoreQueryCoreTest, mapData_ChooseOneTrunk);
		FRIEND_TEST(StoreQueryCoreTest, mapData_ChooseManyTrunks);
	//filterData
		FRIEND_TEST(StoreQueryCoreTest, filterData_AllData);
		FRIEND_TEST(StoreQueryCoreTest, filterData_ExistingTags);
		FRIEND_TEST(StoreQueryCoreTest, filterData_NonExistingTags);
	//selectData
		FRIEND_TEST(StoreQueryCoreTest, ExecuteQuery_SpecificTimeFrame_AllTags_NoAggregation);
		FRIEND_TEST(StoreQueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_NoAggregation);
	//add
		FRIEND_TEST(StoreQueryCoreTest, add_Empty);
		FRIEND_TEST(StoreQueryCoreTest, add_EvenNumberOfValues);
		FRIEND_TEST(StoreQueryCoreTest, add_OddNumberOfValues);
	//average
		FRIEND_TEST(StoreQueryCoreTest, average_Empty);
		FRIEND_TEST(StoreQueryCoreTest, average_Positive);
		FRIEND_TEST(StoreQueryCoreTest, average_Negative);
	//max
		FRIEND_TEST(StoreQueryCoreTest, max_Empty);
		FRIEND_TEST(StoreQueryCoreTest, max_Positive);
		FRIEND_TEST(StoreQueryCoreTest, max_Negative);
	//min
		FRIEND_TEST(StoreQueryCoreTest, min_Empty);
		FRIEND_TEST(StoreQueryCoreTest, min_Positive);
		FRIEND_TEST(StoreQueryCoreTest, min_Negative);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* QUERYCORE_H_ */
