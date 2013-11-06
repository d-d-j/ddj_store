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
		StoreController_Pointer* controller;

		//for(int i=0; i<this->_cudaDevicesCount; i++)
		for(int i=0; i<1; i++)
		{
			controller = new StoreController_Pointer(new store::StoreController(i));
			this->_controllers.insert({i,*controller});
			delete controller;
		}

		this->_client = new Client(&_requestSignal);
	}

	Node::~Node()
	{
		// Disconnect and release client
		delete this->_client;

		// Stop task thread and release it
		{
			boost::mutex::scoped_lock lock(this->_taskMutex);
			this->_taskThread->interrupt();
		}
		this->_taskThread->join();

		delete this->_taskBarrier;
		delete this->_taskThread;
	}

	void Node::CreateTask(taskRequest request)
	{
		// Add a new task to task monitor
		store::StoreTask_Pointer task =
				this->_storeTaskMonitor->AddTask(request.task_id, request.type, request.data);

		// TODO: Run this task in one selected StoreController when Insert or in all StoreControllers otherwise
		this->_controllers[0]->ExecuteTask(task);
	}

	void Node::taskThreadFunction()
	{
		boost::unique_lock<boost::mutex> lock(this->_taskMutex);
		this->_taskBarrier->wait();
		try
		{
			while(1)
			{
				this->_taskCond.wait(lock);

				// Get all compleated tasks
				boost::container::vector<store::StoreTask_Pointer> compleatedTasks =
						this->_storeTaskMonitor->PopCompleatedTasks();

				// Send results of the tasks to master
				int compleatedTaskCount = compleatedTasks.size();
				TaskResult* result;

				for(int i=0; i<compleatedTaskCount; i++)
				{
					if(compleatedTasks[i]->GetType() == SelectAll)
					{
						// Get result of the task
						result = compleatedTasks[i]->GetResult();

						// Send result
						this->_client->SendTaskResult(result);
						// Destroy Task and TaskResult
						delete result;
					}
				}
			}
		}
		catch(boost::thread_interrupted& ex)
		{
			return;
		}
		catch(...)
		{
		}
	}
} /* namespace ddj */
