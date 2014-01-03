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
		// Values aggregation
		None = 0,
		Add = 1,
		Min = 2,
		Max = 3,
		Average = 4,
		StdDeviation = 5,
		Count = 6,
		Variance = 7,
		Differential = 8,
		Integral = 9,
		// Series aggregation
		None_Series = 10,
		Add_Series = 11,
		Min_Series = 12,
		MAx_Series = 13,
		Average_Series = 14,
		StdDeviation_Series = 15,
		Count_Series = 16,
		Variance_Series = 17,
		Differential_Series = 18,
		Integral_Series = 19
	};

} /* namespace store */
} /* namespace ddj */

#endif /* STOREAGGREGATIONTYPE_H_ */
