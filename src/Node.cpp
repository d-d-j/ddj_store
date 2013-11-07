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
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Node constructor [BEGIN]"));

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

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Node constructor [END]"));
	}

	Node::~Node()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Node destructor [BEGIN]"));

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

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Node destructor [END]"));
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
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Task thread [BEGIN]"));

		boost::unique_lock<boost::mutex> lock(this->_taskMutex);

		this->_taskBarrier->wait();
		try
		{
			while(1)
			{
				this->_taskCond.wait(lock);

				// Get all completed tasks
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

						// TODO: only for testing purposes - should be removed
						int n = result->result_size / sizeof(store::storeElement);
						store::storeElement* elements = (store::storeElement*)result->result_data;
						for(int k=0; k<n; k++)
							LOG4CPLUS_DEBUG_FMT(this->_logger, "Select all element[%d]: {tag=%d, metric=%d, time=%llu, value=%f", k, elements[k].metric, elements[k].tag, elements[k].time, elements[k].value);

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
			LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Task thread interrupted [END SUCCESS]"));
			return;
		}
		catch(std::exception& ex)
		{
			LOG4CPLUS_ERROR_FMT(this->_logger, "Task thread failed with exception - [%s] [FAILED]", ex.what());
		}
		catch(...)
		{
			LOG4CPLUS_FATAL(this->_logger, LOG4CPLUS_TEXT("Task thread error with unknown reason [FAILED]"));
			throw;
		}
	}
} /* namespace ddj */
