#ifndef INFOELEMENT_H_
#define INFOELEMENT_H_

#include "../Store/StoreIncludes.h"

namespace ddj {
namespace store {

	/*! \struct infoElement
	 \brief A structure containing the indexes of element trunks in the GPU store array

	 InfoElement structs are stored in B+Tree structure. They contains start and end times of
	 elements from single trunk and it's position in GPU structure array. It is also signed
	 by tag of elements from trunk.
	*/
	typedef struct infoElement
	{
		public:
			/* FIELDS */
			tag_type tag;
			ullint startTime;
			ullint endTime;
			info_value_type startValue;
			info_value_type endValue;
			/* CONSTRUCTORS */
			infoElement(){ tag = 0; startTime = 0; endTime = 0; startValue = 0; endValue = 0; }
			infoElement(tag_type _tag, ullint _startTime, ullint _endTime, info_value_type _startValue, info_value_type _endValue)
			: tag(_tag), startTime(_startTime), endTime(_endTime), startValue(_startValue), endValue(_endValue) {}
			~infoElement() {}
	} infoElement;
	
} /* namespace ddj */
} /* namespace store */

#endif /* defined(INFOELEMENT_H_) */
