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

	StoreTask* StoreTaskMonitor::AddTask(int taskId, TaskType type, void* taskData)
	{
		boost::mutex::scoped_lock lock(this->_mutex);
		StoreTask* newTask = new StoreTask(taskId, type, taskData, this->_condResponseReady);
		this->_tasks.push_back(&newTask);
		this->_taskCount++;
		return newTask;
	}

} /* namespace store */
} /* namespace ddj */
