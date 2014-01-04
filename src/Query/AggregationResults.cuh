#ifndef AGGREGATIONRESULTS_H_
#define AGGREGATIONRESULTS_H_

namespace ddj {
namespace query {
namespace results {

	struct sumResult
	{
		float sum;

		sumResult():sum(0){}
		sumResult(float sum):sum(sum){}
	};

	struct varianceResult
	{
		int32_t count;
		float mean;
		float M2;

		__host__ __device__
		varianceResult():count(0),mean(0),M2(0){}
		__host__ __device__
		varianceResult(int count, float mean, float M2):count(count),mean(mean),M2(M2){}
		__host__ __device__
		varianceResult(const varianceResult& result)
		{
			count = result.count;
			mean = result.mean;
			M2 = result.M2;
		}
	};

	struct averageResult
	{
		float sum;
		int32_t count;

		__host__ __device__
		averageResult():sum(0),count(0){}
		__host__ __device__
		averageResult(float sum, int count):sum(sum),count(count){}
		__host__ __device__
		averageResult(const averageResult& result)
		{
			sum = result.sum;
			count = result.count;
		}
	};

}
}
}
#endif /* AGGREGATIONRESULTS_H_ */
