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
			boost::condition_variable* cond)
	{
		this->_taskId = taskId;
		this->_type = type;
		this->_taskData = taskData;
		this->_condResponseReady = cond;
		this->_isCompleated = false;
		this->_isSuccessfull = false;
		this->_message = NULL;
		this->_resultData = NULL;
		this->_resultSize = 0;
	}

	StoreTask::~StoreTask()
	{
		if(this->_taskData != nullptr)
			free(this->_taskData);
		if(this->_resultData != nullptr)
			free(this->_resultData);
	}

	void StoreTask::SetResult(
			bool isSuccessfull,
			char* message,
			void* resultData,
			int resultSize)
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		if(this->_isCompleated == true)
			throw std::runtime_error("StoreTask::SetResult cannot set new result if another still exists");
		this->_isCompleated = true;
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
				this->_type,
				this->_resultData,
				this->_resultSize
				);
	}

	TaskType StoreTask::GetType()
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		return this->_type;
	}

	void* StoreTask::GetData()
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		return this->_taskData;
	}

	bool StoreTask::IsCompleated()
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		return this->_isCompleated;
	}

} /* namespace store */
} /* namespace ddj */
