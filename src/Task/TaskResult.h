/*
 * TaskResult.h
 *
 *  Created on: 21-09-2013
 *      Author: ghashd
 */

#include "../Store/StoreIncludes.h"
#include "TaskType.h"

#ifndef TASKRESULT_H_
#define TASKRESULT_H_

namespace ddj {

	struct TaskResult
	{
	public:
		int task_id;
		TaskType type;
		void* result_data;
		int result_size;
		bool is_successfull;
		char* message;

		TaskResult(
				int taskId,
				TaskType type,
				bool isSuccessfull,
				char* message = nullptr,
				void* resultData = nullptr,
				int resultSize = 0);

		TaskResult(const TaskResult & result);

		virtual ~TaskResult();
	};

} /* namespace ddj */
#endif /* TASKRESULT_H_ */
