#ifndef TASKREDUCER_H_
#define TASKREDUCER_H_

#include "../Query/Query.h"
#include "../Query/QueryAggregationData.h"

namespace ddj {
namespace task {

class TaskReducer {
public:
	static size_t Reduce(query::Query* query, void* taskResult, size_t resultSize, void** reducedResult);
private:
	static size_t reduceHistogramValue(
			query::data::histogramValueData* data,
			void* taskResult,
			size_t resultSize,
			void** reducedResult);
	static size_t reduceHistogramTime(
			query::data::histogramTimeData* data,
			void* taskResult,
			size_t resultSize,
			void** reducedResult);
};

} /* namespace task */
} /* namespace ddj */
#endif /* TASKREDUCER_H_ */
