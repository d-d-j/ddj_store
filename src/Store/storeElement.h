#ifndef STOREELEMENT_H_
#define STOREELEMENT_H_

#include "storeTypedefs.h"
#include <string>
#include <sstream>
#include <cstring>

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
			int tag;
			metric_type metric;
			ullint time;
			store_value_type value;

			/* CONSTRUCTORS */
			storeElement(){ tag = 0; metric = 0; time = 0; value = 0; }
			storeElement(int _tag, metric_type _metric, ullint _time, store_value_type _value)
			: tag(_tag), metric(_metric), time(_time), value(_value) {}
			storeElement(const storeElement& elem)
			{
				this->tag = elem.tag;
				this->metric = elem.metric;
				this->time = elem.time;
				this->value = elem.value;
			}
			~storeElement(){}

			std::string toString()
			{
				 std::ostringstream stream;
				 stream << "[" << tag << ", " << metric << ", " << time << ", " << value << "]";
			     return  stream.str();
			}
	} storeElement;

} /* namespace ddj */
} /* namespace store */

#endif /* defined( STOREELEMENT_H_) */
