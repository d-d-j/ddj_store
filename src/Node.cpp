/*
 * Node.cpp
 *
 *  Created on: 30-09-2013
 *      Author: ghashd
 */

#include "Node.h"
#include "CUDA/GpuStore.cuh"

namespace ddj
{
	Node::Node()
	{
		this->_storeTaskMonitor = new store::StoreTaskMonitor(&(this->_taskCond));
		this->_taskBarrier = new boost::barrier(2);

		// START TASK THRAED
		this->_taskThread = new boost::thread(boost::bind(&Node::taskThreadFunction, this));
		this->_taskBarrier->wait();

		// connect CreateTaskMethod to _newTaskSignal
		this->_requestSignal.connect(boost::bind(&Node::CreateTask, this, _1));

		this->_cudaDevicesCount = gpuGetCudaDevicesCount();
		pantheios::log_INFORMATIONAL(PSTR("Found "), pantheios::integer(this->_cudaDevicesCount), PSTR(" cuda devices."));
		StoreController_Pointer* controller;

		for(int i=0; i<this->_cudaDevicesCount; i++)
		{
			controller = new StoreController_Pointer(new store::StoreController(i));
			this->_controllers.insert({i,*controller});
			delete controller;
		}

		// TODO: create network client (client constructor should wait until it connects to master)
		this->_client = new Client(&_requestSignal);
	}

	Node::~Node()
	{
		h_LogThreadDebug("Node destructor started");

		// Disconnect and release client
		//delete this->_client;

		// Stop task thread and release it
		{
			boost::mutex::scoped_lock lock(this->_taskMutex);
			h_LogThreadDebug("StoreController locked task's mutex");
			this->_taskThread->interrupt();
		}
		this->_taskThread->join();

		delete this->_taskBarrier;
		delete this->_taskThread;

		h_LogThreadDebug("Node destructor ended");
	}

	void Node::CreateTask(taskRequest request)
	{
		// Add a new task to task monitor
		store::StoreTask_Pointer task =
				this->_storeTaskMonitor->AddTask(request.task_id, request.type, request.data);

		// Run tasks in store controllers TODO: Implement this properly
		this->_controllers[0]->ExecuteTask(task);
	}

	void Node::taskThreadFunction()
	{
		h_LogThreadDebug("Task thread started");
		boost::unique_lock<boost::mutex> lock(this->_taskMutex);
		h_LogThreadDebug("Task thread locked his mutex");
		this->_taskBarrier->wait();
		try
		{
			while(1)
			{
				h_LogThreadDebug("Task thread is waiting");
				this->_taskCond.wait(lock);
				h_LogThreadDebug("Task thread starts his job");

				// Get all compleated tasks
				boost::container::vector<store::StoreTask_Pointer> compleatedTasks =
						this->_storeTaskMonitor->PopCompleatedTasks();

				// TODO: Send results of the tasks to master


				h_LogThreadDebug("Task thread ends his job");
	}
		}
		catch(boost::thread_interrupted& ex)
		{
			h_LogThreadDebug("TaskThread ended as interrupted [Success]");
			return;
		}
		catch(...)
		{
			h_LogThreadDebug("TaskThread ended with error [Failure]");
		}
	}
} /* namespace ddj */
