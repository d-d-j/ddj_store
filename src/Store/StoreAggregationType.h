/*
 * StoreAggregationType.h
 *
 *  Created on: 16-12-2013
 *      Author: ghash
 */

#ifndef STOREAGGREGATIONTYPE_H_
#define STOREAGGREGATIONTYPE_H_

namespace ddj {
namespace store {

	enum AggregationType
	{
		None = 0,
		Add = 1,
		Min = 2,
		Max = 3
	};

} /* namespace store */
} /* namespace ddj */

#endif /* STOREAGGREGATIONTYPE_H_ */
