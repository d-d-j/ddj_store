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

	StoreController::StoreController(int gpuDeviceId)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller constructor [BEGIN]"));

		this->_gpuDeviceId = gpuDeviceId;
		this->_buffers = new Buffers_Map();

		// PREPARE TASK FUNCTIONS DICTIONARY
		this->populateTaskFunctions();

		// CREATE CUDA CONTROLLER (Controlls gpu store side)
		this->_cudaController = new CudaController(STREAMS_NUM_UPLOAD, STREAMS_NUM_QUERY);

		// CREATE GPU UPLOAD MONITOR
		this->_gpuUploadMonitor = new GpuUploadMonitor(this->_cudaController);

		// CREATE QUERY MONITOR
		this->_queryMonitor = new QueryMonitor(this->_cudaController);

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller constructor [END]"));
	}

	StoreController::~StoreController()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller destructor [BEGIN]"));

		delete this->_buffers;
		delete this->_queryMonitor;
		delete this->_gpuUploadMonitor;

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller destructor [BEGIN]"));
	}

	void StoreController::ExecuteTask(StoreTask_Pointer task)
	{
		// Fire a function from _TaskFunctions with this taskId
		this->_taskFunctions[task->GetType()](task);
	}

	void StoreController::populateTaskFunctions()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller - populate task functions [BEGIN]"));

		// INSERT
		_taskFunctions.insert({ Insert, boost::bind(&StoreController::insertTask, this, _1) });

		// SELECT ALL
		_taskFunctions.insert({ SelectAll, boost::bind(&StoreController::selectAllTask, this, _1) });

		// FLUSH
		_taskFunctions.insert({ Flush, boost::bind(&StoreController::flushTask, this, _1) });

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller - populate task functions [END]"));
}

	void StoreController::insertTask(StoreTask_Pointer task)
	{
		// Check possible errors
		if(task == nullptr || task->GetType() != Insert)
		{
			throw std::runtime_error("Error in insertTask function - wrong argument");
		}

		// GET store element from task data
		storeElement* element = (storeElement*)(task->GetData());

		// Log element to insert
		LOG4CPLUS_INFO_FMT(_logger, "Insert task - Insert element[ tag=%d, metric=%d, time=%llu, value=%f", element->tag, element->series, element->time, element->value);

		// Create buffer with element's metric if not exists
		if(!this->_buffers->count(element->tag))
		{
			StoreBuffer_Pointer newBuf(new StoreBuffer(element->tag, this->_gpuUploadMonitor));
			this->_buffers->insert({element->tag, newBuf});
		}

		(*_buffers)[element->tag]->Insert(element);

		// TODO: Check this function for exceptions and errors and set result to error and some error message if failed
		task->SetResult(true, nullptr, nullptr, 0);
	}

	void StoreController::selectAllTask(StoreTask_Pointer task)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("SelectAll task [BEGIN]"));

		// Check possible errors
		if(task == nullptr || task->GetType() != SelectAll)
		{
			LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT("selectAllTask function - wrong argument [FAILED]"));
			throw std::runtime_error("Error in selectAllTask function - wrong argument");
		}

		// Get all data from GPU store
		storeElement* queryResult;

		// TODO: Implement all possible exceptions catching from SelectAll function
		try
		{
			size_t sizeOfResult = this->_queryMonitor->SelectAll(&queryResult);
			task->SetResult(true, nullptr, queryResult, sizeOfResult);
			LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("SelectAll task [END]"));
		}
		catch(std::exception& ex)
		{
			LOG4CPLUS_ERROR_FMT(this->_logger, "SelectAll task error with exception - [%s] [FAILED]", ex.what());
			task->SetResult(false, ex.what(), nullptr, 0);
		}
		catch(...)
		{
			task->SetResult(false, nullptr, nullptr, 0);
			LOG4CPLUS_FATAL(this->_logger, LOG4CPLUS_TEXT("SelectAll task error [FAILED]"));
		}
	}

	void StoreController::flushTask(StoreTask_Pointer task)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Flush task [BEGIN]"));

		// Check possible errors
		if(task == nullptr || task->GetType() != Flush)
		{
			LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT("flushTask function - wrong argument [FAILED]"));
			throw std::runtime_error("Error in flushTask function - wrong argument");
		}

		try
		{
			// Iterate through store buffers and flush them to GPU memory (Sync)
			// TODO: Do it all flushes in parallel but sync them before returning from this function - flushTask should be sync.
			for(Buffers_Map::iterator it = _buffers->begin(); it != _buffers->end(); it++)
			{
				it-> second->Flush();
			}
		}
		catch(std::exception& ex)
		{
			LOG4CPLUS_ERROR_FMT(this->_logger, "Flush task error with exception - [%s] [FAILED]", ex.what());
			task->SetResult(false, ex.what(), nullptr, 0);
		}
		catch(...)
		{
			task->SetResult(false, nullptr, nullptr, 0);
			LOG4CPLUS_FATAL(this->_logger, LOG4CPLUS_TEXT("Flush task error [FAILED]"));
		}

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Flush task [END]"));
	}
} /* namespace store */
} /* namespace ddj */
