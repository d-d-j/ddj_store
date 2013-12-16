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
		int* tp = new int[2*size];
		memcpy(&tp, (char*)queryData+position, 2*size*sizeof(int32_t));
		position+=2*size*sizeof(int32_t);
		while(size--)
			this->timePeriods.push_back(ullintPair{tp[2*size],tp[2*size+1]});

		// Get aggregationType
		int type = 0;
		memcpy(&type, (char*)queryData+position, sizeof(int32_t));
		aggregationType = (task::AggregationType)type;

		delete [] mt;
		delete [] t;
		delete [] tp;
	}

} /* namespace ddj */
} /* namespace store */
