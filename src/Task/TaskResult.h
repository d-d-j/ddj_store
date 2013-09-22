/*
 * TaskResult.h
 *
 *  Created on: 21-09-2013
 *      Author: ghashd
 */

#include "StoreIncludes.h"

#ifndef TASKRESULT_H_
#define TASKRESULT_H_

namespace ddj {

enum TaskType
{
	Insert = 1,
	Select = 2
};

struct TaskResult
{
public:
	int task_id;
	void* result_data;
	int result_size;
	TaskType type;
	bool is_successfull;
	char* message;

	TaskResult(
			int taskId,
			bool isSuccessfull,
			char* message = nullptr,
			void* resultData = nullptr,
			int resultSize = 0);

	TaskResult(const TaskResult & result);

	virtual ~TaskResult();
};

} /* namespace ddj */
#endif /* TASKRESULT_H_ */
