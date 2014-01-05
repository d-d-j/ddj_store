#ifndef QUERYAGGREGATION_H_
#define QUERYAGGREGATION_H_

#include "AggregationType.h"
#include "../Store/StoreElement.cuh"
#include "../Cuda/CudaAggregation.cuh"
#include <gtest/gtest.h>
#include <boost/unordered_map.hpp>
#include <boost/function.hpp>
#include <boost/bind.hpp>

namespace ddj {
namespace query {

	using namespace store;

	typedef boost::container::vector<ullintPair> ullintPairVector;

	typedef boost::function<size_t (
									storeElement* elements,
									size_t size,
									void** result,
									ullintPairVector*
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

			size_t sum(storeElement* elements, size_t dataSize, void** result, ullintPairVector* dataLocationInfo = nullptr);
			size_t min(storeElement* elements, size_t dataSize, void** result, ullintPairVector* dataLocationInfo = nullptr);
			size_t max(storeElement* elements, size_t dataSize, void** result, ullintPairVector* dataLocationInfo = nullptr);
			size_t average(storeElement* elements, size_t dataSize, void** result, ullintPairVector* dataLocationInfo = nullptr);
			size_t stdDeviation(storeElement* elements, size_t dataSize, void** result, ullintPairVector* dataLocationInfo = nullptr);
			size_t variance(storeElement* elements, size_t dataSize, void** result, ullintPairVector* dataLocationInfo = nullptr);
			size_t differential(storeElement* elements, size_t dataSize, void** result, ullintPairVector* dataLocationInfo = nullptr);
			size_t integral(storeElement* elements, size_t dataSize, void** result, ullintPairVector* dataLocationInfo = nullptr);

			friend class QueryAggregationTest;

			/*********/
			/* TESTS */
			/*********/

			//sum
				FRIEND_TEST(QueryAggregationTest, sum_Empty);
				FRIEND_TEST(QueryAggregationTest, sum_EvenNumberOfValues);
				FRIEND_TEST(QueryAggregationTest, sum_OddNumberOfValues);
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
	};

} /* namespace query */
} /* namespace ddj */
#endif /* QUERYAGGREGATION_H_ */
