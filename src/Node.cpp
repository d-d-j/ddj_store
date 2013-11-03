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
		h_LogThreadDebug("Node destructor started");

		// Disconnect and release client
		delete this->_client;

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

		// TODO: Run this task in one selected StoreController when Insert or in all StoreControllers otherwise
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

				// Send results of the tasks to master
				int compleatedTaskCount = compleatedTasks.size();
				TaskResult* result;
				for(int i=0; i<compleatedTaskCount; i++)
				{
					if(compleatedTasks[i]->GetType() == SelectAll)
					{
						// Get result of the task
						result = compleatedTasks[i]->GetResult();

						/*	FOR TESTING - SHOULD BE REMOVED
 						store::storeElement* data;
						size_t size;
						int count;
						data = (store::storeElement*)result->result_data;
						size = result->result_size;
						count = size / sizeof(store::storeElement);
						for(int j=0; j<count; j++)
							printf("SelectAll result[%d]: t:%d s:%d time:%d value:%f",
									j, data[j].tag, data[j].series, (int)data[j].time, data[j].value);
						 */

						// Send result
						this->_client->SendTaskResult(result);
						// Destroy Task and TaskResult
						delete result;
					}
				}

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
