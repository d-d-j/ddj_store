#ifndef STOREELEMENT_H_
#define STOREELEMENT_H_

#include "storeTypedefs.h"
#include <string>
#include <sstream>

namespace ddj {
namespace store {

	/*!	\struct storeElement
	 	\brief this structs will be stored in GPU side array where all data will be stored.

	 	storeElement is the data structure send to this system by customers as a single
	 	time series event. It contains time as unsigned long long int, is tagged, has
	 	a value and belongs to certain series.
	 */
	typedef struct storeElement
	{
		public:
			/* FIELDS */
			int series;
			tag_type tag;
			ullint time;
			store_value_type value;

			/* CONSTRUCTORS */
			storeElement(){ series = 0; tag = 0; time = 0; value = 0; }
			storeElement(int _series, tag_type _tag, ullint _time, store_value_type _value)
			: series(_series), tag(_tag), time(_time), value(_value) {}
			storeElement(const storeElement& elem)
			{
				this->series = elem.series;
				this->tag = elem.tag;
				this->time = elem.time;
				this->value = elem.value;
			}
			~storeElement(){}

			std::string toString()
			{
				 std::ostringstream stream;
			     stream << "["<<series<<", "<<tag<<", "<<time<<", "<<value<<"]";
			     return  stream.str();
			}
	} storeElement;

} /* namespace ddj */
} /* namespace store */

#endif /* defined( STOREELEMENT_H_) */
