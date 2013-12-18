#include "StoreQuery.h"

namespace ddj {
namespace store {

	storeQuery::storeQuery(void* queryData)
	{
		int position = 0;
		int size = 0;

		// Get size
		memcpy(&size, (char*)queryData+position, sizeof(int32_t));
		position+=sizeof(int32_t);

		// Get metrices
		metric_type* mt = new metric_type[size];
		memcpy(mt, (char*)queryData+position, size*sizeof(metric_type));
		position+=size*sizeof(metric_type);
		while(size--)
			this->metrices.push_back(mt[size]);

		// Get size
		memcpy(&size, (char*)queryData+position, sizeof(int32_t));
		position+=sizeof(int32_t);

		// Get tags
		int* t = new int[size];
		memcpy(t, (char*)queryData+position, size*sizeof(int));
		position+=size*sizeof(int);
		while(size--)
			this->tags.push_back(t[size]);

		// Get size
		memcpy(&size, (char*)queryData+position, sizeof(int32_t));
		position+=sizeof(int32_t);

		// Get timePeriods
		ullint* tp = new ullint[2*size];
		memcpy(&tp, (char*)queryData+position, 2*size*sizeof(ullint));
		position+=2*size*sizeof(ullint);
		while(size--)
			this->timePeriods.push_back(ullintPair{tp[2*size],tp[2*size+1]});

		// Get aggregationType
		int type = 0;
		memcpy(&type, (char*)queryData+position, sizeof(int32_t));
		aggregationType = (AggregationType)type;

		delete [] mt;
		delete [] t;
		delete [] tp;
	}

	std::string storeQuery::toString()
	{
		 std::ostringstream stream;

		 stream << "query[";
		 stream << "aggregationType: " << this->aggregationType;
		 stream << "; metrices:";
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
} /* namespace store */
