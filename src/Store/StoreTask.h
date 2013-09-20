/*
 * StoreTask.h
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#ifndef STORETASK_H_
#define STORETASK_H_

#include "../Store/StoreIncludes.h"

namespace ddj {
namespace store {

enum StoreResultType
{
	InsertResult = 1
};

class StoreTask {
private:
	void* _result;
	StoreResultType _type;
	int _dataPartsCount;
	boost::condition_variable* _condResponseReady;
	boost::mutex _mutex;
public:
	StoreTask(boost::condition_variable* cond, StoreResultType type, int partsCount = 1);
	virtual ~StoreTask();

	void AddResult(void* result);
	void* GetResult();
};


} /* namespace store */
} /* namespace ddj */
#endif /* STORETASK_H_ */
