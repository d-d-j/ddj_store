#ifndef AGGREGATIONTYPE_H_
#define AGGREGATIONTYPE_H_

namespace ddj {

	enum AggregationType
	{
		// Values aggregation
		None = 0,
		Sum = 1,
		Min = 2,
		Max = 3,
		Average = 4,
		StdDeviation = 5,
		Variance = 6,
		Differential = 7,
		Integral = 8,
		// Series aggregation
		Sum_Series = 11,
		Min_Series = 12,
		MAx_Series = 13,
		Average_Series = 14,
		StdDeviation_Series = 15,
		Variance_Series = 16,
		Differential_Series = 17,
		Integral_Series = 18
	};

} /* namespace ddj */

#endif /* AGGREGATIONTYPE_H_ */
