#ifndef STOREQUERY_H_
#define STOREQUERY_H_

#include "StoreAggregationType.h"
#include "../Core/UllintPair.h"
#include <boost/container/vector.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <sstream>

namespace ddj {
namespace store {

	struct storeQuery
	{
		boost::container::vector<metric_type> metrices;
		boost::container::vector<int> tags;
		boost::container::vector<ullintPair> timePeriods;
		task::AggregationType aggregationType;

		storeQuery(void* queryData);

		std::string toString();
	};

} /* namespace ddj */
} /* namespace store */

#endif /* STOREQUERY_H_ */
