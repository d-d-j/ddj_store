/*
 * TaskRequest.h
 *
 *  Created on: 28-09-2013
 *      Author: ghashd
 */

#ifndef TASKREQUEST_H_
#define TASKREQUEST_H_

#include "TaskType.h"
#include <string>
#include <sstream>

namespace ddj {
namespace task {

	struct taskRequest
	{
	public:
		int64_t task_id;	/**< id set for a task by master */
		TaskType type;	/**< task type, for example Insert */
		int32_t size;
		int32_t device_id;
		// data is released in StoreTask
		void* data;		/**< data for a task, for example Select data or elem to insert */

		taskRequest():task_id(0),type(Error),size(0),device_id(TASK_ALL_DEVICES),data(nullptr){}
		taskRequest(int64_t id, TaskType type, int32_t device_id, int32_t data_size, void* data)
		:task_id(id),type(type),size(data_size),device_id(device_id),data(data){}
		taskRequest(const taskRequest& request)
		{
			task_id = request.task_id;
			type = request.type;
			data = request.data;
			size = request.size;
			device_id = request.device_id;
		}

		std::string toString()
		{
			 std::ostringstream stream;
			 stream << "[" << task_id << ", " << type << ", " << size << "]";
			 return  stream.str();
		}
	};

} /* namespace store */
} /* namespace ddj */

#endif /* TASKREQUEST_H_ */
