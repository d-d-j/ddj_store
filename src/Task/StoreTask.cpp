/*
 * StoreTask.cpp
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#include "StoreTask.h"

namespace ddj {
namespace store {

StoreTask::StoreTask(
		int taskId,
		TaskType type,
		void* taskData,
		int dataSize,
		boost::condition_variable* cond)
{
	this->_taskId = taskId;
	this->_type = type;
	this->_taskData = taskData;
	this->_dataSize = dataSize;
	this->_condResponseReady = cond;
	this->_isDone = false;
	this->_isSuccessfull = false;
	this->_message = NULL;
	this->_resultData = NULL;
	this->_resultSize = 0;
}

StoreTask::~StoreTask()
{
	free(this->_taskData);
	free(this->_resultData);
}

void StoreTask::SetResult(
		bool isSuccessfull,
		char* message,
		void* resultData,
		int resultSize)
{
	boost::lock_guard<boost::mutex> guard(this->_mutex);
	if(this->_isDone == true)
		throw std::runtime_error("StoreTask::SetResult cannot set new result if another still exists");
	this->_isDone = true;
	this->_isSuccessfull = isSuccessfull;
	this->_message = message;
	this->_resultData = resultData;
	this->_resultSize = resultSize;
	this->_condResponseReady->notify_one();
}

TaskResult* StoreTask::GetResult()
{
	boost::lock_guard<boost::mutex> guard(this->_mutex);
	return new TaskResult(
			this->_taskId,
			this->_isSuccessfull,
			this->_message,
			this->_resultData,
			this->_resultSize
			);
}

} /* namespace store */
} /* namespace ddj */
