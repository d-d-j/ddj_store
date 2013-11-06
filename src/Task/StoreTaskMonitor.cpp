/*
 * StoreTaskMonitor.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "StoreTaskMonitor.h"

namespace ddj {
namespace store {

	StoreTaskMonitor::StoreTaskMonitor(boost::condition_variable* condResponseReady)
	{
		this->_condResponseReady = condResponseReady;
		this->_taskCount = 0;
	}

	StoreTaskMonitor::~StoreTaskMonitor()
	{
	}

	StoreTask_Pointer StoreTaskMonitor::AddTask(int taskId, TaskType type, void* taskData)
	{
		boost::mutex::scoped_lock lock(this->_mutex);

		StoreTask_Pointer newTask(new StoreTask(taskId, type, taskData, this->_condResponseReady));
		this->_tasks.push_back(newTask);
		this->_taskCount++;
		return newTask;
	}

	boost::container::vector<StoreTask_Pointer> StoreTaskMonitor::PopCompleatedTasks()
	{
		boost::mutex::scoped_lock lock(this->_mutex);

		// Copy compleated tasks to result
		boost::container::vector<StoreTask_Pointer> result(this->_tasks.size());
		auto it = std::copy_if(
				this->_tasks.begin(),
				this->_tasks.end(),
				result.begin(),
				[](StoreTask_Pointer task){return task->IsCompleated();}
				);
		result.resize(std::distance(result.begin(),it));

		// Remove compleated tasks from _tasks
		std::remove_if(
				this->_tasks.begin(),
				this->_tasks.end(),
				[](StoreTask_Pointer task){return task->IsCompleated();}
				);

		// Return result
		return result;
	}

} /* namespace store */
} /* namespace ddj */
