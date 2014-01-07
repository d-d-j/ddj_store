#ifndef AGGREGATIONTYPE_H_
#define AGGREGATIONTYPE_H_

namespace ddj {

	enum AggregationType
	{
		None = 0,
		Sum = 1,
		Min = 2,
		Max = 3,
		Average = 4,
		StdDeviation = 5,
		Variance = 6,
		Differential = 7,
		Integral = 8,
		Histogram_Value = 9,
		Histogram_Time = 10,
		Skewness = 11,
		Kurtosis = 12
	};

} /* namespace ddj */

#endif /* AGGREGATIONTYPE_H_ */
