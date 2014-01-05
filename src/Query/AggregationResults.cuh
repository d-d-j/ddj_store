#ifndef AGGREGATIONRESULTS_H_
#define AGGREGATIONRESULTS_H_

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

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

		HOST DEVICE
		varianceResult():count(0),mean(0),M2(0){}
		HOST DEVICE
		varianceResult(int count, float mean, float M2):count(count),mean(mean),M2(M2){}
		HOST DEVICE
		varianceResult(const varianceResult& result)
		{
			count = result.count;
			mean = result.mean;
			M2 = result.M2;
		}
	};

	struct averageResult
	{
		int32_t count;
		float sum;

		HOST DEVICE
		averageResult():count(0), sum(0) {}
		HOST DEVICE
		averageResult(float sum, int count):count(count), sum(sum) {}
		HOST DEVICE
		averageResult(const averageResult& result)
		{
			sum = result.sum;
			count = result.count;
		}
	};

	struct integralResult
	{
		float integral;
		// left store element
		float left_value;
		int64_t left_time;
		// right store element
		float right_value;
		int64_t right_time;
	};

}
}
}
#endif /* AGGREGATIONRESULTS_H_ */
