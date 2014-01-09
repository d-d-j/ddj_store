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
		this->_query = nullptr;
	}

	Task::~Task()
	{
		boost::mutex::scoped_lock lock(this->_mutex);

		delete static_cast<char*>(this->_taskData);
		delete this->_message;
		free(this->_resultData);
		delete this->_result;
		delete this->_query;
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
		boost::mutex::scoped_lock lock(this->_mutex);

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
			// REDUCE TASK RESULTS

			if(this->_query != nullptr)
			{
				void* reducedResult;
				size_t newResultSize =
						TaskReducer::Reduce(this->_query, this->_resultData, this->_resultSize, &reducedResult);
				if(reducedResult != nullptr)
				{
					delete this->_resultData;
					this->_resultData = reducedResult;
					this->_resultSize = newResultSize;
				}
			}

			// SET TASK RESULT
			this->_result = new taskResult(
					this->_taskId,
					this->_type,
					this->_resultData,
					this->_resultSize
					);
			this->_isCompleated = true;
			this->_condResponseReady->notify_one();
		}
	}

	void Task::SetQuery(query::Query* query)
	{
		boost::mutex::scoped_lock lock(this->_mutex);
		this->_query = query;
	}

	taskResult* Task::GetResult()
	{
		boost::mutex::scoped_lock lock(this->_mutex);
		return this->_result;
	}

	TaskType Task::GetType()
	{
		boost::mutex::scoped_lock lock(this->_mutex);
		return this->_type;
	}

	void* Task::GetData()
	{
		boost::mutex::scoped_lock lock(this->_mutex);
		return this->_taskData;
	}

	bool Task::IsCompleated()
	{
		boost::mutex::scoped_lock lock(this->_mutex);
		return this->_isCompleated;
	}

	int64_t Task::GetId()
	{
		boost::mutex::scoped_lock lock(this->_mutex);
		return this->_taskId;
	}

} /* namespace task */
} /* namespace ddj */
