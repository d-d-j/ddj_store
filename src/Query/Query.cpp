#include "Query.h"

namespace ddj {
namespace query {

	Query::Query(void* queryData)
	{
		int position = 0;
		int size = 0;

		// Get size
		memcpy(&size, (char*)queryData+position, sizeof(int32_t));
		position+=sizeof(int32_t);

		// Get metrics
		metric_type* mt = (metric_type*)((char*)queryData+position);
		position+=size*sizeof(metric_type);
		while(size--)
			this->metrices.push_back(mt[size]);

		// Get size
		memcpy(&size, (char*)queryData+position, sizeof(int32_t));
		position+=sizeof(int32_t);

		// Get tags
		int* t = (int*)((char*)queryData+position);
		position+=size*sizeof(int);
		while(size--)
			this->tags.push_back(t[size]);

		// Get size
		memcpy(&size, (char*)queryData+position, sizeof(int32_t));
		position+=sizeof(int32_t);

		// Get timePeriods
		ullint* tp = (ullint*)((char*)queryData+position);
		position+=2*size*sizeof(ullint);
		while(size--)
			this->timePeriods.push_back(ullintPair{tp[2*size],tp[2*size+1]});

		// Get aggregationType
		int type = 0;
		memcpy(&type, (char*)queryData+position, sizeof(int32_t));
		aggregationType = (AggregationType)type;
	}

	std::string Query::toString()
	{
		 std::ostringstream stream;

		 stream << "query[";
		 stream << "aggregationType: " << this->aggregationType;
		 stream << "; metrics:";
		 BOOST_FOREACH(metric_type &m, this->metrices)
		 {
			 stream << " " << m;
		 }
		 stream << "; tags:";
		 BOOST_FOREACH(int &t, this->tags)
		 {
			 stream << " " << t;
		 }
		 stream << "; timePeriods:";
		 BOOST_FOREACH(ullintPair &tp, this->timePeriods)
		 {
			 stream << " " << tp.toString();
		 }
		 stream << "]";
		 return  stream.str();
	}

} /* namespace ddj */
} /* namespace query */
