#ifndef AGGREGATIONDATA_H_
#define AGGREGATIONDATA_H_

namespace ddj {
namespace query {
namespace data {

	struct histogramData
	{
		float min;
		float max;
		int32_t bucketCount;

		public:
			histogramData(float min, float max, int32_t bucketCount)
				:min(min),max(max),bucketCount(bucketCount){}
			histogramData(const histogramData& rhs)
			{
				min = rhs.min;
				max = rhs.max;
				bucketCount = rhs.bucketCount;
			}
			~histogramData(){}
	};

}
}
}
#endif /* AGGREGATIONDATA_H_ */
