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
		int result_size;
		void* result_data;

		TaskResult(
				int taskId,
				TaskType type,
				void* resultData = nullptr,
				int resultSize = 0);

		TaskResult(const TaskResult & result);

		virtual ~TaskResult();
	};

} /* namespace ddj */
#endif /* TASKRESULT_H_ */
