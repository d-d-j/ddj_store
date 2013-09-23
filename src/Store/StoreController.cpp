/*
 *  StoreController.cpp
 *  StoreController
 *
 *  Created by Karol Dzitkowski on 27.07.2013.
 *  Copyright (c) 2013 Karol Dzitkowski. All rights reserved.
 *
 *      NAZEWNICTWO
 * 1. nazwy klas:  CamelStyle z dużej litery np. StoreController
 * 2. nazwy struktur camelStyle z małej litery np. storeElement
 * 3. nazwy pól prywatnych camelStyle z małej litery z podkreśleniem _backBuffer
 * 4. nazwy pól publicznych i zmiennych globalnych słowa rozdzielamy _ i z małych liter np. memory_available
 * 5. define z dużych liter i rozdzielamy _ np. BUFFER_SIZE
 * 6. nazwy metod publicznych z dużej litery CamelStyle np. InsertValue() oraz parametry funkcji z małych liter camelStyle np. InsertValue(int valToInsert);
 * 7. nazwy metod prywatnych z małej litery camelStyle
 * 8. nazwy funkcji "prywatnych" w plikach cpp z małej litery z _ czyli, insert_value(int val_to_insert);
 * 9. nazwy funkcji globalnych czyli w plikach .h najczęściej inline h_InsertValue() dla funkcji na CPU g_InsertValue() dla funkcji na GPU
 */

#include "StoreController.h"

namespace ddj {
namespace store {

	StoreController::StoreController()
	{
		h_LogThreadDebug("StoreController constructor started");

		this->_buffers = new boost::unordered_map<tag_type, StoreBuffer_Pointer>();
		this->_taskBarrier = new boost::barrier(2);
		this->_storeTaskMonitor = new StoreTaskMonitor(&(this->_taskCond));

		// START TASK THRAED
		this->_taskThread = new boost::thread(boost::bind(&StoreController::taskThreadFunction, this));
		this->_taskBarrier->wait();

		// PREPARE TASK FUNCTIONS DICTIONARY
		this->populateTaskFunctions();

		h_LogThreadDebug("StoreController constructor ended");
	}

	StoreController::~StoreController()
	{
		h_LogThreadDebug("StoreController destructor started");

		// STOP TASK THREAD
		{
			boost::mutex::scoped_lock lock(this->_taskMutex);
			h_LogThreadDebug("StoreController locked task's mutex");
			this->_taskThread->interrupt();
		}
		this->_taskThread->join();

		delete this->_buffers;
		delete this->_taskBarrier;
		delete this->_taskThread;

		h_LogThreadDebug("StoreController destructor ended");
	}

	void StoreController::CreateTask(int taskId, TaskType type, void* taskData)
	{
		// Add a new task to task monitor
		StoreTask* task = this->_storeTaskMonitor->AddTask(taskId, type, taskData);
		// Fire a function from _TaskFunctions with this taskId
		this->_taskFunctions[type](task);
	}

	void StoreController::taskThreadFunction()
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
				h_LogThreadDebug("Task thread is doing his job");
				// TODO: Implement taskThread real job...
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

	void StoreController::populateTaskFunctions()
	{
		std::pair<int, taskFunc> pair;

		// INSERT
		pair.first = 1;
		pair.second = boost::bind(&StoreController::insertTask, this, _1);
		_taskFunctions.insert(pair);
	}

	void StoreController::insertTask(StoreTask* task)
	{
		h_LogThreadDebug("Insert task function started");

		// Check possible errors
		if(task == nullptr || task->GetType() != Insert)
		{
			h_LogThreadError("Error in insertTask function - wrong argument");
			throw std::runtime_error("Error in insertTask function - wrong argument");
		}

		// GET store element from task data
		storeElement* element = (storeElement*)(task->GetData());

		// GET buffer with element's tag or create one if not exists
		if(this->_buffers->count(element->tag))	// if such a buffer exists
		{
			(*_buffers)[element->tag]->Insert(element);
		}
		else
		{
			StoreBuffer_Pointer newBuf(new StoreBuffer(element->tag, &(this->_gpuUploadMonitor)));
			this->_buffers->insert({element->tag, newBuf});

		}

		h_LogThreadDebug("Insert task function ended");
	}

} /* namespace store */
} /* namespace ddj */
