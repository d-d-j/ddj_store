#ifndef AGGREGATIONRESULTS_H_
#define AGGREGATIONRESULTS_H_

namespace ddj {
namespace query {
namespace results {

	struct stdDeviationResult
	{
		int32_t count;
		float mean;
		float M2;

		bool operator== (const stdDeviationResult& rhs) const
		{
			if(count == rhs.count && mean == rhs.mean && M2 == rhs.M2)
				return true;
			else return false;
		}
	};

	struct averageResult
	{
		double sum;
		int32_t count;

		bool operator== (const averageResult& rhs) const
		{
			if(count == rhs.count && sum == rhs.sum)
				return true;
			else return false;
		}
	};

}
}
}
#endif /* AGGREGATIONRESULTS_H_ */
