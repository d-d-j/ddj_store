/*
 * StoreTask.h
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#ifndef STORETASK_H_
#define STORETASK_H_

#include "TaskResult.h"
#include <boost/thread.hpp>
#include <boost/utility.hpp>

namespace ddj {
namespace task {

class Task : public boost::noncopyable
{
private:
	/* TASK */
	int _taskId;
	TaskType _type;
	void* _taskData;

	/* RESULT */
	void* _resultData;
	int _resultSize;
	char* _message;
	bool _isSuccessfull;
	bool _isCompleated;

	/* MONITOR */
	boost::condition_variable* _condResponseReady;
	boost::mutex _mutex;

public:
	Task(
			int taskId,
			TaskType type,
			void* taskData,
			boost::condition_variable* cond);

	virtual ~Task();

	void SetResult(
			bool isSuccessfull,
			const char* message = nullptr,
			void* resultData = nullptr,
			int resultSize = 0);

	taskResult* GetResult();

	/* GETTERS */
	TaskType GetType();
	void* GetData();
	bool IsCompleated();
};

typedef boost::shared_ptr<Task> Task_Pointer;

} /* namespace store */
} /* namespace ddj */
#endif /* STORETASK_H_ */
