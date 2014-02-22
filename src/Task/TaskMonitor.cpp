/*
 * StoreTaskMonitor.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "TaskMonitor.h"

namespace ddj {
namespace task {

	TaskMonitor::TaskMonitor(boost::condition_variable *condResponseReady)
	{
	    this->_condResponseReady = condResponseReady;
	    this->_taskCount = 0;
	}

	TaskMonitor::~TaskMonitor()
	{
	}

	Task_Pointer TaskMonitor::AddTask(int taskId, TaskType type, void *taskData, int expectedResultCount)
	{
	    boost::mutex::scoped_lock lock(this->_mutex);

	    Task_Pointer newTask(new Task(taskId, type, taskData, expectedResultCount, this->_condResponseReady));
	    this->_tasks.push_back(newTask);
	    this->_taskCount++;
	    return newTask;
	}

	boost::container::vector<Task_Pointer> TaskMonitor::PopCompletedTasks()
	{
	    boost::mutex::scoped_lock lock(this->_mutex);

	    // Copy completed tasks to result
	    boost::container::vector<Task_Pointer> result(_tasks.size());
	    auto it = std::copy_if(
          _tasks.begin(),
          _tasks.end(),
          result.begin(),
          [](Task_Pointer task){ return task->IsCompleted(); }
     	);
	    result.resize(std::distance(result.begin(),it));

	    // Remove completed tasks from tasks
	    auto comparator = [](const Task_Pointer &x, const Task_Pointer &y){ return x->GetId() < y->GetId(); };
		std::sort(result.begin(), result.end(), comparator);
		std::sort(_tasks.begin(), _tasks.end(), comparator);

		boost::container::vector<Task_Pointer> difference(_tasks.size());
		it = std::set_difference(
			_tasks.begin(), _tasks.end(),
			result.begin(), result.end(),
			difference.begin(),
			comparator
		);

		difference.resize(std::distance(difference.begin(),it));

		_tasks = difference;

	    // Return result
	    return result;
	}

} /* namespace store */
} /* namespace ddj */
