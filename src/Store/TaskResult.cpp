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
					bool isSuccessfull,
					char* message,
					void* resultData,
					int resultSize
				):
			task_id(taskId),
			result_data(resultData),
			result_size(resultSize),
			is_successfull(isSuccessfull),
			message(message) {}

TaskResult::~TaskResult()
{
	free(this->message);
	free(this->result_data);
}

} /* namespace ddj */
