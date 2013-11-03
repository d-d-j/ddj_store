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

		TaskResult(const TaskResult & result)
		{
			task_id = result.task_id;
			type = result.type;
			result_size = result.result_size;
			result_data = result.result_data;
		}

		virtual ~TaskResult();

		std::string toString()
		{
			 std::ostringstream stream;
		     stream << "["<<task_id<<", "<<type<<", "<<result_size<<"]";
		     return  stream.str();
		}
	};

} /* namespace ddj */
#endif /* TASKRESULT_H_ */
