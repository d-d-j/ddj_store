#ifndef TASKREDUCER_H_
#define TASKREDUCER_H_

#include "../Query/Query.h"
#include "../Query/AggregationData.h"

namespace ddj {
namespace task {

class TaskReducer {
public:
	static void Reduce(query::Query* query, void* taskResult, int resultCount, void** reducedResult);
private:
	static void reduceHistogramValue(query::data::histogramValueData* data, void* taskResult, int resultCount, void** reducedResult);
	static void reduceHistogramTime(query::data::histogramTimeData* data, void* taskResult, int resultCount, void** reducedResult);
};

} /* namespace task */
} /* namespace ddj */
#endif /* TASKREDUCER_H_ */
