#ifndef QUERY_H_
#define QUERY_H_

#include "AggregationType.h"
#include "../Core/UllintPair.h"
#include <boost/container/vector.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <sstream>

namespace ddj {
namespace query {

	struct Query : public boost::noncopyable
	{
		boost::container::vector<metric_type> metrices;
		boost::container::vector<int> tags;
		boost::container::vector<ullintPair> timePeriods;
		AggregationType aggregationType;
		void* aggregationData;

		Query():aggregationType(None),aggregationData(NULL){}
		Query(void* queryData);
		~Query(){}

		std::string toString();
	};

} /* namespace ddj */
} /* namespace query */

#endif /* QUERY_H_ */
