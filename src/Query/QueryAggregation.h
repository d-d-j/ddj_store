#ifndef QUERYAGGREGATION_H_
#define QUERYAGGREGATION_H_

#include "AggregationType.h"
#include "AggregationData.h"
#include "../Store/StoreElement.cuh"
#include "../Cuda/CudaAggregation.cuh"
#include <gtest/gtest.h>
#include <boost/unordered_map.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <vector_types.h>

namespace ddj {
namespace query {

	using namespace store;

	typedef boost::container::vector<ullintPair> ullintPairVector;

	typedef boost::function<size_t (
									storeElement* elements,
									size_t size,
									void** result,
									Query*
									)
							> aggregationFunc;

	class QueryAggregation {
		public:
			QueryAggregation();
			virtual ~QueryAggregation();

		protected:
			boost::unordered_map<int, aggregationFunc> _aggregationFunctions;

		private:
			void propagateAggregationMethods();

			size_t sum(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t min(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t max(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t average(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t stdDeviation(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t variance(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t skewness(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t kurtosis(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t differential(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t integral(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t histogramValue(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t histogramTime(storeElement* elements, size_t dataSize, void** result, Query* query);
			size_t sumSeries(storeElement* elements, size_t dataSize, void** result, Query* query);

			friend class QueryAggregationTest;

			/*********/
			/* TESTS */
			/*********/

			//sum
				FRIEND_TEST(QueryAggregationTest, sum_Empty);
				FRIEND_TEST(QueryAggregationTest, sum_EvenNumberOfValues);
				FRIEND_TEST(QueryAggregationTest, sum_OddNumberOfValues);
				FRIEND_TEST(QueryAggregationTest, sum_OneElement);
			//min
				FRIEND_TEST(QueryAggregationTest, min_Empty);
				FRIEND_TEST(QueryAggregationTest, min_Positive);
				FRIEND_TEST(QueryAggregationTest, min_Negative);
			//max
				FRIEND_TEST(QueryAggregationTest, max_Empty);
				FRIEND_TEST(QueryAggregationTest, max_Positive);
				FRIEND_TEST(QueryAggregationTest, max_Negative);
			//average
				FRIEND_TEST(QueryAggregationTest, average_Empty);
				FRIEND_TEST(QueryAggregationTest, average_Linear);
				FRIEND_TEST(QueryAggregationTest, average_Sinusoidal);
			//stdDeviation or Variance
				FRIEND_TEST(QueryAggregationTest, stdDeviationOrVariance_Empty);
				FRIEND_TEST(QueryAggregationTest, stdDeviationOrVariance_Simple);
				FRIEND_TEST(QueryAggregationTest, stdDeviationOrVariance_Linear);
			//differential
			//integral
				FRIEND_TEST(QueryAggregationTest, integral_Empty);
				FRIEND_TEST(QueryAggregationTest, integral_Simple_OneTrunk);
				FRIEND_TEST(QueryAggregationTest, integral_Simple_OneTrunk_SingleElement);
				FRIEND_TEST(QueryAggregationTest, integral_Simple_ManyTrunks_EqualTrunks);
			//histogram on Value
				FRIEND_TEST(QueryAggregationTest, histogram_Value_Empty);
				FRIEND_TEST(QueryAggregationTest, histogram_Value_Simple_4Buckets);
				FRIEND_TEST(QueryAggregationTest, histogram_Value_Simple_1Bucket);
				FRIEND_TEST(QueryAggregationTest, histogram_Value_ValuesOnBucketsEdges_LeftInclusive_4Buckets);
				FRIEND_TEST(QueryAggregationTest, histogram_Value_ValuesOnBucketsEdges_RightExclusive_4Buckets);
			//histogram on Time
				FRIEND_TEST(QueryAggregationTest, histogram_Time_Empty);
				FRIEND_TEST(QueryAggregationTest, histogram_Time_Simple_4Buckets);
				FRIEND_TEST(QueryAggregationTest, histogram_Time_Simple_1Bucket);
				FRIEND_TEST(QueryAggregationTest, histogram_Time_ValuesOnBucketsEdges_LeftInclusive_4Buckets);
				FRIEND_TEST(QueryAggregationTest, histogram_Time_ValuesOnBucketsEdges_RightExclusive_4Buckets);
			//series sum
				FRIEND_TEST(QueryAggregationTest, series_Sum_Empty);
				FRIEND_TEST(QueryAggregationTest, series_Sum_WrongQuery_NoTimePeriods);
				FRIEND_TEST(QueryAggregationTest, series_Sum_WrongQuery_NoTags);
				FRIEND_TEST(QueryAggregationTest, series_Sum_WrongQuery_NoMetrics);
				FRIEND_TEST(QueryAggregationTest, series_Sum_Simple_3tags1metric_EqualValues_ConsistentTimeIntervals);
				FRIEND_TEST(QueryAggregationTest, series_Sum_Simple_3tags3metrics_LinearValues_ConsistentTimeIntervals);
				FRIEND_TEST(QueryAggregationTest, series_Sum_Simple_3tags1metric_LinearValues_InterpolationNeeded);
				FRIEND_TEST(QueryAggregationTest, series_Sum_Normal_2tags1metrics_SinCosValues_ConsistentTimeIntervals);
				FRIEND_TEST(QueryAggregationTest, series_Sum_Normal_2tags1metrics_SinCosValues_InterpolationNeeded);
	};

} /* namespace query */
} /* namespace ddj */
#endif /* QUERYAGGREGATION_H_ */
