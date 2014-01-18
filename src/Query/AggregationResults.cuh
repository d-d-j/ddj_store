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
		float m2;

		HOST DEVICE
		varianceResult():count(0),mean(0),m2(0){}
		HOST DEVICE
		varianceResult(int count, float mean, float M2):count(count),mean(mean),m2(M2){}
		HOST DEVICE
		varianceResult(const varianceResult& result)
		{
			count = result.count;
			mean = result.mean;
			m2 = result.m2;
		}
	};

	struct skewnessResult
	{
		int32_t count;
		float mean;
		float m2;
		float m3;

		HOST DEVICE
		skewnessResult():count(0),mean(0),m2(0),m3(0){}
		HOST DEVICE
		skewnessResult(int count, float mean, float M2, float M3):count(count),mean(mean),m2(M2),m3(M3){}
		HOST DEVICE
		skewnessResult(const skewnessResult& result)
		{
			count = result.count;
			mean = result.mean;
			m2 = result.m2;
			m3 = result.m3;
		}
	};

	struct kurtosisResult
	{
		int32_t count;
		float mean;
		float m2;
		float m3;
		float m4;

		HOST DEVICE
		kurtosisResult():count(0),mean(0),m2(0),m3(0),m4(0){}
		HOST DEVICE
		kurtosisResult(int count, float mean, float M2, float M3, float M4):count(count),mean(mean),m2(M2),m3(M3),m4(M4){}
		HOST DEVICE
		kurtosisResult(const kurtosisResult& result)
		{
			count = result.count;
			mean = result.mean;
			m2 = result.m2;
			m3 = result.m3;
			m4 = result.m4;
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
