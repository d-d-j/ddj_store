/*
 * Node.cpp
 *
 *  Created on: 30-09-2013
 *      Author: ghashd
 */

#include "Node.h"

namespace ddj
{
	Node::Node() : _logger(Logger::getRoot()), _config(Config::GetInstance())
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Node constructor [BEGIN]"));

		this->_taskMonitor = new task::TaskMonitor(&(this->_taskCond));
		this->_taskBarrier = new boost::barrier(2);

		// START TASK THRAED
		this->_taskThread = new boost::thread(boost::bind(&Node::taskThreadFunction, this));
		this->_taskBarrier->wait();

		// connect CreateTaskMethod to _newTaskSignal
		this->_requestSignal.connect(boost::bind(&Node::CreateTask, this, _1));

		this->_cudaDevicesCount = _cudaCommons.CudaGetDevicesCountAndPrint();

		StoreController_Pointer* controller;

		//for(int i=0; i<this->_cudaDevicesCount; i++)
		for(int i=0; i<1; i++)
		{
			// Check if GPU i satisfies ddj_store requirements
			if(!_cudaCommons.CudaCheckDeviceForRequirements(i)) continue;
			controller = new StoreController_Pointer(new store::StoreController(i));
			this->_controllers.insert({i,*controller});
			delete controller;
		}

		//throw exception if suitable cuda gpu devices count == 0
		if(this->_controllers.empty())
		{
			std::string errString = "!! NO CUDA DEVICE !!";
			errString.append(" - there is no cuda device connected or it does not satisfy ddj_store requirements...");
			throw std::runtime_error(errString);
		}

		// CONNECT TO MASTER AND LOG IN
		this->_client = new network::NetworkClient(&_requestSignal);
		int* devices = new int[1];
		devices[0] = 0;
		network::networkLoginRequest* loginRequest = new network::networkLoginRequest(devices, 1);
		this->_client->SendLoginRequest(loginRequest);
		delete [] devices;
		delete loginRequest;

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

	void Node::CreateTask(task::taskRequest request)
	{
		// Add a new task to task monitor
		task::Task_Pointer task =
				this->_taskMonitor->AddTask(request.task_id, request.type, request.data);

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
				boost::container::vector<task::Task_Pointer> compleatedTasks =
						this->_taskMonitor->PopCompleatedTasks();

				// Send results of the tasks to master
				int compleatedTaskCount = compleatedTasks.size();
				task::taskResult* result;

				for(int i=0; i<compleatedTaskCount; i++)
				{
					if(compleatedTasks[i]->GetType() == task::Select)
					{
						// Get result of the task
						result = compleatedTasks[i]->GetResult();

						// TODO: only for testing purposes - should be removed
						int n = result->result_size / sizeof(store::storeElement);
						store::storeElement* elements = (store::storeElement*)result->result_data;
						for(int k=0; k<n; k++)
							LOG4CPLUS_DEBUG_FMT(this->_logger, "Select element[%d]: {metric=%d, tag=%d, time=%llu, value=%f}", k, elements[k].metric, elements[k].tag, elements[k].time, elements[k].value);

						// Send result
						this->_client->SendTaskResult(result);

						// Destroy Task and TaskResult
						delete result;
					}
					else if(compleatedTasks[i]->GetType() == task::Info)
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
