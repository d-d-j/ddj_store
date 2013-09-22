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
	int _taskId;
	void* _resultData;
	int _resultSize;
	TaskType _type;
	char* _message;
	bool _isSuccessfull;
	bool _isDone;
	boost::condition_variable* _condResponseReady;
	boost::mutex _mutex;

public:
	StoreTask(int taskId, boost::condition_variable* cond, TaskType type);
	virtual ~StoreTask();

	void SetResult(
			bool isSuccessfull,
			char* message = nullptr,
			void* resultData = nullptr,
			int resultSize = 0);

	TaskResult* GetResult();
};

} /* namespace store */
} /* namespace ddj */
#endif /* STORETASK_H_ */
