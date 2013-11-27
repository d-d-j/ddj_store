/*
 * StoreTaskMonitor.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "StoreTaskMonitor.h"

namespace ddj {
namespace task {

	TaskMonitor::TaskMonitor(boost::condition_variable* condResponseReady)
	{
		this->_condResponseReady = condResponseReady;
		this->_taskCount = 0;
	}

	TaskMonitor::~TaskMonitor()
	{
	}

	StoreTask_Pointer TaskMonitor::AddTask(int taskId, TaskType type, void* taskData)
	{
		boost::mutex::scoped_lock lock(this->_mutex);

		StoreTask_Pointer newTask(new Task(taskId, type, taskData, this->_condResponseReady));
		this->_tasks.push_back(newTask);
		this->_taskCount++;
		return newTask;
	}

	boost::container::vector<StoreTask_Pointer> TaskMonitor::PopCompleatedTasks()
	{
		boost::mutex::scoped_lock lock(this->_mutex);

		// Copy compleated tasks to result
		boost::container::vector<StoreTask_Pointer> result(this->_tasks.size());
		auto it = std::copy_if(
				this->_tasks.begin(),
				this->_tasks.end(),
				result.begin(),
				[](StoreTask_Pointer task){ if(task != nullptr) return task->IsCompleated(); else return false; }
				);
		result.resize(std::distance(result.begin(),it));

		// Remove compleated tasks from _tasks
		std::remove_if(
				this->_tasks.begin(),
				this->_tasks.end(),
				[](StoreTask_Pointer task){ if(task != nullptr) return task->IsCompleated(); else return false; }
				);

		// Return result
		return result;
	}

} /* namespace store */
} /* namespace ddj */
