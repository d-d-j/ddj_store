/*
 * StoreTaskMonitor.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "TaskMonitor.h"

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

	Task_Pointer TaskMonitor::AddTask(int taskId, TaskType type, int32_t deviceId, void* taskData)
	{
		boost::mutex::scoped_lock lock(this->_mutex);

		Task_Pointer newTask(new Task(taskId, type, deviceId, taskData, this->_condResponseReady));
		this->_tasks.push_back(newTask);
		this->_taskCount++;
		return newTask;
	}

	boost::container::vector<Task_Pointer> TaskMonitor::PopCompleatedTasks()
	{
		boost::mutex::scoped_lock lock(this->_mutex);

		// Copy completed tasks to result
		boost::container::vector<Task_Pointer> result(this->_tasks.size());
		auto it = std::copy_if(
				this->_tasks.begin(),
				this->_tasks.end(),
				result.begin(),
				[](Task_Pointer task){ if(task != nullptr) return task->IsCompleated(); else return false; }
				);
		result.resize(std::distance(result.begin(),it));

		// Remove completed tasks from tasks
		_tasks.erase( std::remove_if(
				this->_tasks.begin(),
				this->_tasks.end(),
				[](Task_Pointer task){ if(task != nullptr) return task->IsCompleated(); else return false; }
				) );

		// Return result
		return result;
	}

} /* namespace store */
} /* namespace ddj */
