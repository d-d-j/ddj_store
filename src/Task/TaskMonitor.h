/*
 * StoreTaskMonitor.h
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#ifndef STORETASKMONITOR_H_
#define STORETASKMONITOR_H_

#include "../Store/StoreIncludes.h"
#include "StoreTask.h"
#include "TaskType.h"

namespace ddj {
namespace task {

	class TaskMonitor
	{
		/* FIELDS */
		public:
			boost::container::vector<StoreTask_Pointer> _tasks;
			boost::condition_variable* _condResponseReady;
			boost::mutex _mutex;
			volatile sig_atomic_t _taskCount;

		/* METHODS */
		public:
			TaskMonitor(boost::condition_variable* condResponseReady);
			virtual ~TaskMonitor();
			StoreTask_Pointer AddTask(int taskId, TaskType type, void* taskData);
			boost::container::vector<StoreTask_Pointer> PopCompleatedTasks();
	};

} /* namespace store */
} /* namespace ddj */

#endif /* STORETASKMONITOR_H_ */
