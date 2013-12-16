/*
 * StoreAggregationType.h
 *
 *  Created on: 16-12-2013
 *      Author: ghash
 */

#ifndef STOREAGGREGATIONTYPE_H_
#define STOREAGGREGATIONTYPE_H_

namespace ddj {
namespace task {

	enum AggregationType
	{
		None = 0,
		Add = 1,
		Average = 2
	};

} /* namespace task */
} /* namespace ddj */

#endif /* STOREAGGREGATIONTYPE_H_ */