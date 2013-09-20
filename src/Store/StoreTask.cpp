/*
 * StoreTask.cpp
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#include "StoreTask.h"

namespace ddj {
namespace store {

StoreTask::StoreTask(boost::condition_variable* cond, StoreResultType type, int partsCount)
{
	boost::lock_guard<boost::mutex> guard(_mutex);
	_dataPartsCount = partsCount;
	_condResponseReady = cond;
	_type = type;
}

StoreTask::~StoreTask()
{
	if(_type == InsertResult)
		delete (int*)_result;

}

void StoreTask::AddResult(void* result)
{
	boost::lock_guard<boost::mutex> guard(_mutex);
	_dataPartsCount--;
	if(_type == InsertResult)
		_result = result;
}

void* StoreTask::GetResult()
{
	boost::lock_guard<boost::mutex> guard(_mutex);
	if(_type == InsertResult)
		return _result;
	else return NULL;
}

} /* namespace store */
} /* namespace ddj */
