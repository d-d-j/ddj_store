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
		this->_message = nullptr;
		this->_resultData = nullptr;
		this->_resultSize = 0;
		this->_currentResultCount = 0;
		this->_expectedResultCount = expectedResultCount;
		this->_result = nullptr;
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
		if (this->_message != nullptr)
		{
			delete this->_message;
		}
		this->_taskData = nullptr;
		this->_resultData = nullptr;
		this->_message = nullptr;
	}

	void Task::appendMessage(const char* message)
	{
		size_t lengthA = 0, lengthB = 0, length;
		if (this->_message != nullptr)
		{
			lengthA = std::char_traits<char>::length(this->_message);
		}
		if (message != nullptr)
		{
			lengthB = std::char_traits<char>::length(message);
		}

		length = lengthA + lengthB + 1;

		char *result = new char[length];
		snprintf(result, length, "%s%s", this->_message ? this->_message : "", message ? message : "");

		delete this->_message;
		this->_message = result;
	}

	void Task::SetResult(
			bool isSuccessfull,
			const char* message,
			void* resultData,
			size_t resultSize)
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		// Append message to task message
		this->appendMessage(message);
		this->_isSuccessfull &= isSuccessfull;

		// Append data to task data
		if(resultData != nullptr && resultSize != 0)
		{
			int newSize = this->_resultSize+resultSize;
			void* newTaskResult = malloc(newSize);
			if(this->_resultData != nullptr && this->_resultSize != 0)
			{
				memcpy(newTaskResult, this->_resultData, this->_resultSize);
				free(this->_resultData);
			}
			memcpy((char*)newTaskResult+this->_resultSize, resultData, resultSize);
			this->_resultData = newTaskResult;
			this->_resultSize = newSize;
		}

		// Decide if task is completed and should be reduced
		this->_currentResultCount++;
		if(this->_currentResultCount == this->_expectedResultCount)
		{
			// TODO: REDUCE TASK RESULTS

			// SET TASK RESULT
			this->_result = new taskResult(
					this->_taskId,
					this->_type,
					this->_resultData,
					this->_resultSize,
					this->_message
					);
			this->_isCompleated = true;
			this->_condResponseReady->notify_one();
		}
	}

	taskResult* Task::GetResult()
	{
		boost::lock_guard<boost::mutex> guard(this->_mutex);

		return this->_result;
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

} /* namespace task */
} /* namespace ddj */
