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
	int task_id;	/**< id nadane zadaniu przez mastera */
	TaskType type;	/**< typ zadania np. insert */
	void* data;		/**< dane dla zadania, np. parametry zapytania lub element jeśli np. typ == Insert */
	taskRequest(int id, TaskType type, void* data):task_id(id),type(type),data(data){}
	~taskRequest(){ free(data); }
	taskRequest(const taskRequest& request)
	{
		task_id = request.task_id;
		type = request.type;
		data = request.data;
	}
};

#endif /* TASKREQUEST_H_ */
