/*
 * StoreTask.cpp
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#include "Task.h"

namespace ddj {
namespace task {

	Task::Task(
			int taskId,
			TaskType type,
			void* taskData,
			int expectedResultCount,
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
		this->_currentResultCount = 0;
		this->_expectedResultCount = expectedResultCount;
	}

	Task::~Task()
	{
		if(this->_taskData != nullptr)
		{
			delete (char*)this->_taskData;
		}
		if(this->_resultData != nullptr)
		{
			free(this->_resultData);
		}
	}

	void Task::appendMessage(char* message)
	{
		std::string a(this->_message);
		std::string b(message);
		this->_message = (char*)(a+b).c_str();
	}

	void Task::SetResult(
			bool isSuccessfull,
			const char* message,
			void* resultData,
			size_t resultSize)
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		// Append message to task message
		this->appendMessage(this->_message);
		this->_isSuccessfull &= isSuccessfull;

		// Append data to task data
		void* newTaskResult = malloc(this->_resultSize+resultSize);
		memcpy(newTaskResult, this->_resultData, sizeof(this->_resultSize));
		free(this->_resultData);
		memcpy((char*)newTaskResult+this->_resultSize, resultData, resultSize);
		free(resultData);
		this->_resultData = newTaskResult;
		this->_resultSize += resultSize;

		// Decide if task is completed and should be reduced
		this->_currentResultCount++;
		if(this->_currentResultCount == this->_expectedResultCount)
		{
			// TODO: REDUCE TASK RESULTS

			this->_isCompleated = true;
			this->_condResponseReady->notify_one();
		}
	}

	taskResult* Task::GetResult()
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		return new taskResult(
				this->_taskId,
				this->_type,
				this->_resultData,
				this->_resultSize
				);
	}

	TaskType Task::GetType()
	{
		return this->_type;
	}

	void* Task::GetData()
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		return this->_taskData;
	}

	bool Task::IsCompleated()
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		return this->_isCompleated;
	}

} /* namespace store */
} /* namespace ddj */
