#ifndef AGGREGATIONDATA_H_
#define AGGREGATIONDATA_H_

#include <cstdlib>
#include "../Store/StoreTypedefs.h"

namespace ddj {
namespace query {
namespace data {

	struct histogramValueData
	{
		float min;
		float max;
		int32_t bucketCount;

	public:
		histogramValueData(float min, float max, int32_t bucketCount)
			:min(min),max(max),bucketCount(bucketCount){}
		histogramValueData(const histogramValueData& rhs)
		{
			min = rhs.min;
			max = rhs.max;
			bucketCount = rhs.bucketCount;
		}
		~histogramValueData(){}
	};

	struct histogramTimeData
	{
		ullint min;
		ullint max;
		int32_t bucketCount;

	public:
		histogramTimeData(ullint min, ullint max, int32_t bucketCount)
				:min(min),max(max),bucketCount(bucketCount){}
		histogramTimeData(const histogramValueData& rhs)
		{
			min = rhs.min;
			max = rhs.max;
			bucketCount = rhs.bucketCount;
		}
		~histogramTimeData(){}
	};
}
}
}
#endif /* AGGREGATIONDATA_H_ */
