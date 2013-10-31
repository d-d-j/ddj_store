/*
 * TaskRequest.h
 *
 *  Created on: 28-09-2013
 *      Author: ghashd
 */

#ifndef TASKREQUEST_H_
#define TASKREQUEST_H_

#include "TaskType.h"

struct taskRequest
{
public:
	int task_id;	/**< id set for a task by master */
	TaskType type;	/**< task type, for example Insert */
	// data is released in StoreTask
	void* data;		/**< data for a task, for example Select data or elem to insert */

	taskRequest(int id, TaskType type, void* data):task_id(id),type(type),data(data){}
	taskRequest(const taskRequest& request)
	{
		task_id = request.task_id;
		type = request.type;
		data = request.data;
	}
};

#endif /* TASKREQUEST_H_ */
