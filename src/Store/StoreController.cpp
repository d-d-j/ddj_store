/*
 *  StoreController.cpp
 *  StoreController
 *
 *  Created by Karol Dzitkowski on 27.07.2013.
 *  Copyright (c) 2013 Karol Dzitkowski. All rights reserved.
 *
 *      NAZEWNICTWO
 * 1. nazwy klas:  CamelStyle z duÅ¼ej litery np. StoreController
 * 2. nazwy struktur camelStyle z maÅ‚ej litery np. storeElement
 * 3. nazwy pÃ³l prywatnych camelStyle z maÅ‚ej litery z podkreÅ›leniem _backBuffer
 * 4. nazwy pÃ³l publicznych i zmiennych globalnych sÅ‚owa rozdzielamy _ i z maÅ‚ych liter np. memory_available
 * 5. define z duÅ¼ych liter i rozdzielamy _ np. BUFFER_SIZE
 * 6. nazwy metod publicznych z duÅ¼ej litery CamelStyle np. InsertValue() oraz parametry funkcji z maÅ‚ych liter camelStyle np. InsertValue(int valToInsert);
 * 7. nazwy metod prywatnych z maÅ‚ej litery camelStyle
 * 8. nazwy funkcji "prywatnych" w plikach cpp z maÅ‚ej litery z _ czyli, insert_value(int val_to_insert);
 * 9. nazwy funkcji globalnych czyli w plikach .h najczÄ™Å›ciej inline h_InsertValue() dla funkcji na CPU g_InsertValue() dla funkcji na GPU
 */

#include "StoreController.h"

namespace ddj {
namespace store {

	StoreController::StoreController(int gpuDeviceId)
		: _logger(Logger::getRoot()), _config(Config::GetInstance())
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller constructor [BEGIN]"));

		this->_gpuDeviceId = gpuDeviceId;

		this->_buffers = new Buffers_Map();

		// PREPARE TASK FUNCTIONS DICTIONARY
		this->populateTaskFunctions();

		// CREATE CUDA CONTROLLER (Controlls gpu store side)
		this->_cudaController = new CudaController(_config->GetIntValue("STREAMS_NUM_UPLOAD"), _config->GetIntValue("STREAMS_NUM_QUERY"), gpuDeviceId);

		// CREATE STORE QUERY CORE
		this->_queryCore = new StoreQueryCore(this->_cudaController);

		// CREATE STORE UPLOAD CORE
		this->_uploadCore = new StoreUploadCore(this->_cudaController);

		// SET THREAD POOL SIZES
		this->_taskThreadPool.size_controller().resize(this->_config->GetIntValue("THREAD_POOL_SIZE"));

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller constructor [END]"));
	}

	StoreController::~StoreController()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller destructor [BEGIN]"));

		delete this->_buffers;
		delete this->_cudaController;

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller destructor [BEGIN]"));
	}

	void StoreController::ExecuteTask(task::Task_Pointer task)
	{
		// Sechedule a function from _TaskFunctions with this taskId
		task::TaskType type = task->GetType();
		this->_taskThreadPool.schedule(boost::bind(this->_taskFunctions[type], task));
	}

	void StoreController::populateTaskFunctions()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller - populate task functions [BEGIN]"));

		// INSERT
		_taskFunctions.insert({ task::Insert, boost::bind(&StoreController::insertTask, this, _1) });

		// SELECT ALL
		_taskFunctions.insert({ task::Select, boost::bind(&StoreController::selectTask, this, _1) });

		// FLUSH
		_taskFunctions.insert({ task::Flush, boost::bind(&StoreController::flushTask, this, _1) });

		// FLUSH
		_taskFunctions.insert({ task::Info, boost::bind(&StoreController::infoTask, this, _1) });

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller - populate task functions [END]"));
	}

	void StoreController::insertTask(task::Task_Pointer task)
	{
		// Check possible errors
		if(task == nullptr || task->GetType() != task::Insert)
		{
			throw std::runtime_error("Error in insertTask function - wrong argument");
		}

		// GET store element from task data
		storeElement* element = (storeElement*)(task->GetData());

		// Create buffer with element's metric if not exists
		{
			boost::mutex::scoped_lock lock(this->_buffersMutex);
			if(!this->_buffers->count(element->metric))
			{
				StoreBuffer_Pointer newBuf(new StoreBuffer(element->metric, this->_config->GetIntValue("STORE_BUFFER_CAPACITY"), this->_uploadCore));
				this->_buffers->insert({element->metric, newBuf});
			}
		}
		// Log element to insert
		LOG4CPLUS_DEBUG_FMT(_logger, "Insert task - Insert element[ metric=%d, tag=%d, time=%llu, value=%f", element->metric, element->tag, element->time, element->value);
		(*_buffers)[element->metric]->Insert(element);

		// TODO: Check this function for exceptions and errors and set result to error and some error message if failed
		task->SetResult(true, nullptr, nullptr, 0);
	}

	void StoreController::selectTask(task::Task_Pointer task)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("SelectAll task [BEGIN]"));

		// Check possible errors
		if(task == nullptr || task->GetType() != task::Select)
		{
			LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT("selectAllTask function - wrong argument [FAILED]"));
			throw std::runtime_error("Error in selectTask function - wrong argument");
		}

		try
		{
			// Create query from task data
			storeQuery* query = new storeQuery(task->GetData());
			LOG4CPLUS_INFO(this->_logger, "Select task - " << query->toString());

			boost::container::vector<ullintPair>* dataLocationInfo = new boost::container::vector<ullintPair>();

			// if query should be filtered ask StoreBuffer for data location on GPU
			if(query->metrices.size())
			{
				BOOST_FOREACH(metric_type &m, query->metrices)
				{
					boost::container::vector<ullintPair>* locations = (*_buffers)[m]->Select(query->timePeriods);
					dataLocationInfo->insert(dataLocationInfo->end(), locations->begin(), locations->end());
					delete locations;
				}
			}
			// Execute query with optional data locations using StoreQueryCore


			// Set task result and return


		}
		catch(std::exception& ex)
		{
			LOG4CPLUS_ERROR_FMT(this->_logger, "Select task error with exception - [%s] [FAILED]", ex.what());
			task->SetResult(false, ex.what(), nullptr, 0);
		}
		catch(...)
		{
			task->SetResult(false, nullptr, nullptr, 0);
			LOG4CPLUS_FATAL(this->_logger, LOG4CPLUS_TEXT("Select task error [FAILED]"));
		}
	}

	void StoreController::flushTask(task::Task_Pointer task)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Flush task [BEGIN]"));

		// Check possible errors
		if(task == nullptr || task->GetType() != task::Flush)
		{
			LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT("flushTask function - wrong argument [FAILED]"));
			throw std::runtime_error("Error in flushTask function - wrong argument");
		}

		try
		{
			for(Buffers_Map::iterator it = _buffers->begin(); it != _buffers->end(); ++it)
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

	void StoreController::infoTask(task::Task_Pointer task)
	{

	}

} /* namespace store */
} /* namespace ddj */
