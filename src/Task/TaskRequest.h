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

struct taskRequest
{
public:
	int task_id;	/**< id set for a task by master */
	TaskType type;	/**< task type, for example Insert */
	int size;
	// data is released in StoreTask
	void* data;		/**< data for a task, for example Select data or elem to insert */

	taskRequest():task_id(0),type(Error),size(0),data(nullptr){}
	taskRequest(int id, TaskType type, int data_size, void* data)
	:task_id(id),type(type),size(data_size),data(data){}
	taskRequest(const taskRequest& request)
	{
		task_id = request.task_id;
		type = request.type;
		data = request.data;
		size = request.size;
	}

	std::string toString()
	{
		 std::ostringstream stream;
		stream << "[" << task_id << ", " << type << ", " << size << "]";
	     return  stream.str();
	}
};

#endif /* TASKREQUEST_H_ */
