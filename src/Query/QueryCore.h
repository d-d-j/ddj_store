#ifndef QUERYCORE_H_
#define QUERYCORE_H_

#include "Query.h"
#include "QueryAggregation.h"
#include "QueryFilter.cuh"
#include "../Compression/Compression.h"
#include "../Store/StoreElement.cuh"
#include "../Core/Logger.h"
#include "../Cuda/CudaController.h"
#include "../Cuda/CudaIncludes.h"
#include <boost/foreach.hpp>


namespace ddj {
namespace query {

	using namespace store;

	/**
	 * A class used to execute a query on records stored in database on GPU side.
	 * A class with only one public method ExecuteQuery which maps data from main
	 * memory on GPU used by DB. Mapped data is decompressed and then filtered and
	 * aggregated with a function defined by aggregation type stored in query.
	 * @see ExecuteQuery()
	 */
	class QueryCore : public boost::noncopyable, public QueryAggregation
	{
	private:
		CudaController* _cudaController;
		bool _enableCompression;

		/* LOGGER & CONFIG */
		Logger _logger;
		Config* _config;

	public:
		/**
		* A constructor with CudaController pointer as only parameter.
		* It initializes logger and config and sets a pointer to cuda controller.
		* @param cudaController a pointer to CudaController used to access
		* memory on GPU side used by database, and for getting CUDA streams.
		*/
		QueryCore(CudaController* cudaController);

		/**
		* A virtual empty destructor.
		*/
		virtual ~QueryCore();

		/**
		 * Method executing query with optional selected dataLocations.
		 * If no dataLocation provided query is executed to all data in store.
		 * Data is firstly mapped from main DB memory on GPU, then data is decompressed, filtered and aggregated.
		 * @see mapData()
		 * @see decompressData()
		 * @see filterData()
		 * @see aggregateData()
		 * @param queryResult [out]	result of query is returned to this parameter
		 * @param query [in] a pointer to query to execute
		 * @param dataLocationInfo [in] locations of data to use in query
		 * @return size of data produced by query
		 */
		size_t ExecuteQuery(void** queryResult, Query* query, boost::container::vector<ullintPair>* dataLocationInfo = nullptr);

	private:
		/***************************/
		/* DATA MANAGEMENT METHODS */
		/***************************/

		/**
		 * Method aggregates store elements (on device).
		 * It aggregates elements of dataSize size using aggregation defined in query's aggregation
		 * type and return aggregated value to result on host. Store Elements should be already on device.
		 * If aggregation type is None, method copies elements from device to aggregation result on host.
		 * If dataSize equals 0 method should set aggregation result on host to null and returns 0.
		 * @param elements [in] an array of elements on device side to aggregate
		 * @param dataSize [in] a size of an array to aggregate
		 * @param query [in] a pointer to a query
		 * @param result [out] aggregated data is returned as a pointer to data on host.
		 * @return returns size of result data
		 */
		size_t aggregateData(storeElement* elements, size_t dataSize, Query* query, void** result);

		/**
		 * Method mapping all data stored in GPU to selected data locations.
		 * Selects only the fragments of mainGpuArray specified by ullintPair structs stored in optional
		 * parameter dataLocationInfo, for example:
		 * If dataLocationInfo contains ullintPair (100,300) then a part or mainGpuArray
		 * from index 100 to index 300 (including 300) will be returned as new gpu (device) array in data parameter.
		 * If optional dataLocationInfo parameter is not provided, then all data stored in DB is returned.
		 * @param data [out] mapped data is returned as device array in this parameter.
		 * @param dataLocationInfo [in, out] (optional) a pointer to a boost vector of ullintPair structures.
		 * @return returns size of mapped data
		 */
		size_t mapData(void** data, boost::container::vector<ullintPair>* dataLocationInfo = nullptr);

		/**
		 * Method filtering an array of storeElements on GPU by tags and time.
		 * It does nothing if no tags and no time periods are specified in query.
		 * Otherwise, it moves all elements with tag equal to any of provided in query
		 * and with time intersecting one of time periods provided in query
		 * to the front of elements array, and returns a number of these elements.
		 * @param elements [in,out] an array of store elements to filter
		 * @param dataSize [in] a size of elements array
		 * @param query [in] a pointer to the Query
		 * @param dataLocationInfo [in, out] (optional) a boost vector to ullintPair structures
		 * @return number of elements moved to front (number of elements with good tag and time)
		 */
		size_t filterData(storeElement* elements, size_t dataSize, Query* query,
				boost::container::vector<ullintPair>* dataLocationInfo = nullptr);

		/**
		 * Method used to decompress data read from main DB memory on GPU.
		 * It uses Compress class object to decompress data.
		 * @param data [in] a pointer to a compressed memory on device to decompress it.
		 * @param size [in] a size of data to decompress
		 * @param elements [out] a pointer to an array of storeElements that data will be decompressed to.
		 * @param dataLocationInfo [in, out] a boost vector to ullintPair structures
		 * @return a size of decompressed data
		 */
		size_t decompressData(void* data, size_t size, storeElement** elements,
				boost::container::vector<ullintPair>* dataLocationInfo);

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
	//selectData with compression
		FRIEND_TEST(QueryCoreTest, ExecuteQuery_ManyTimeFrames_SpecifiedTags_SumAggregation_OneTrunk_Compression);
	};

} /* namespace query */
} /* namespace ddj */
#endif /* QUERYCORE_H_ */
