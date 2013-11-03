/*
 * TaskResult.cpp
 *
 *  Created on: 21-09-2013
 *      Author: ghashd
 */

#include "TaskResult.h"

namespace ddj {

TaskResult::TaskResult
				(
					int taskId,
					TaskType type,
					void* resultData,
					int resultSize
				):
			task_id(taskId),
			type(type),
			result_data(resultData),
			result_size(resultSize) {}

TaskResult::~TaskResult()
{
	//free(this->result_data);
	//this->result_data = nullptr;
}

} /* namespace ddj */
