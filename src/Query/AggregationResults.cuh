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

	struct statisticResult
	{
		int32_t count;
		float mean;
		float factor;

		HOST DEVICE
		statisticResult():count(0),mean(0),factor(0){}
		HOST DEVICE
		statisticResult(int count, float mean, float M2):count(count),mean(mean),factor(M2){}
		HOST DEVICE
		statisticResult(const statisticResult& result)
		{
			count = result.count;
			mean = result.mean;
			factor = result.factor;
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
