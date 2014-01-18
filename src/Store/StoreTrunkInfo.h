#ifndef INFOELEMENT_H_
#define INFOELEMENT_H_

#include "StoreTypedefs.h"

namespace ddj {
namespace store {

	/*! \struct storeTrunkInfo
	 \brief A structure containing the indexes of element trunks in the GPU store array

	 storeTrunkInfo structs are stored in B+Tree structure. They contains start and end times of
	 elements from single trunk and it's position in GPU structure array. It is also signed
	 by metric of elements from trunk.
	*/
	typedef struct storeTrunkInfo
	{
		public:
		/* FIELDS */
			metric_type metric;
			ullint startTime;
			ullint endTime;
			ullint startValue;
			ullint endValue;

		/* CONSTRUCTORS & DESTRUCTOR */
			storeTrunkInfo(){ metric = 0; startTime = 0; endTime = 0; startValue = 0; endValue = 0; }
			storeTrunkInfo(metric_type _metric, ullint _startTime, ullint _endTime, ullint _startValue, ullint _endValue)
			: metric(_metric), startTime(_startTime), endTime(_endTime), startValue(_startValue), endValue(_endValue) {}
			~storeTrunkInfo() {}

	} storeTrunkInfo;
	
} /* namespace ddj */
} /* namespace store */

#endif /* defined(INFOELEMENT_H_) */
