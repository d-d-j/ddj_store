/*
 * StoreTask.h
 *
 *  Created on: 19-09-2013
 *      Author: ghashd
 */

#ifndef STORETASK_H_
#define STORETASK_H_

#include "TaskResult.h"

namespace ddj {
namespace store {

class StoreTask : public boost::noncopyable
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
	StoreTask(
			int taskId,
			TaskType type,
			void* taskData,
			boost::condition_variable* cond);

	virtual ~StoreTask();

	void SetResult(
			bool isSuccessfull,
			char* message = nullptr,
			void* resultData = nullptr,
			int resultSize = 0);

	TaskResult* GetResult();

	/* GETTERS */
	TaskType GetType();
	void* GetData();
	bool IsCompleated();
};

} /* namespace store */
} /* namespace ddj */
#endif /* STORETASK_H_ */
