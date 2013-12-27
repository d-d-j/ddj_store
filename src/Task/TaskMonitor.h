/*
 * StoreTaskMonitor.h
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#ifndef STORETASKMONITOR_H_
#define STORETASKMONITOR_H_

#include "Task.h"
#include "TaskType.h"
#include <boost/shared_ptr.hpp>
#include <boost/container/vector.hpp>
#include <boost/thread.hpp>
#include <stdlib.h>
#include <algorithm>

namespace ddj {
namespace task {

	class TaskMonitor
	{
		/* FIELDS */
		public:
			int _taskCount;
			boost::container::vector<Task_Pointer> _tasks;
			boost::condition_variable* _condResponseReady;
			boost::mutex _mutex;

		/* METHODS */
		public:
			TaskMonitor(boost::condition_variable* condResponseReady);
			virtual ~TaskMonitor();
			Task_Pointer AddTask(int taskId, TaskType type, int32_t deviceId, void* taskData);
			boost::container::vector<Task_Pointer> PopCompleatedTasks();
	};

} /* namespace store */
} /* namespace ddj */

#endif /* STORETASKMONITOR_H_ */
