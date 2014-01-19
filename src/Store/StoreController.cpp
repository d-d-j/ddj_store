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
		this->_maxLocationsPerJob =
				_config->GetIntValue("MAX_JOB_MEMORY_SIZE") /
				(_config->GetIntValue("STORE_BUFFER_CAPACITY")*sizeof(storeElement));

		LOG4CPLUS_DEBUG_FMT(this->_logger,
				LOG4CPLUS_TEXT("Store controller %d -> max locations per job = %d"),
				gpuDeviceId,
				_maxLocationsPerJob);

		this->_buffers = new Buffers_Map();

		// CREATE CUDA CONTROLLER (Controlls gpu store side)
		this->_cudaController = new CudaController(_config->GetIntValue("STREAMS_NUM_UPLOAD"), _config->GetIntValue("STREAMS_NUM_QUERY"), gpuDeviceId);

		// CREATE STORE QUERY CORE
		this->_queryCore = new QueryCore(this->_cudaController);

		// CREATE STORE UPLOAD CORE
		this->_uploadCore = new StoreUploadCore(this->_cudaController);

		// CREATE STORE INFO CORE
		this->_infoCore = new StoreInfoCore(this->_cudaController);

		// SET THREAD POOL SIZES
		this->_insertThreadPool.size_controller().resize(this->_config->GetIntValue("INSERT_THREAD_POOL_SIZE"));
		this->_selectThreadPool.size_controller().resize(this->_config->GetIntValue("SELECT_THREAD_POOL_SIZE"));

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller constructor [END]"));
	}

	StoreController::~StoreController()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller destructor [BEGIN]"));

		// remove all pending tasks
		_insertThreadPool.clear();
		_selectThreadPool.clear();

		// wait untill all active tasks are finished
		_insertThreadPool.wait();
		_selectThreadPool.wait();

		delete this->_buffers;
		delete this->_uploadCore;
		delete this->_cudaController;

		this->_buffers = nullptr;
		this->_cudaController = nullptr;

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Store controller destructor [BEGIN]"));
	}

	void StoreController::ExecuteTask(task::Task_Pointer task)
	{
		// Sechedule a function from _TaskFunctions with this taskId
		task::TaskType type = task->GetType();
		switch(type)
		{
			case task::Insert:
				this->_insertThreadPool.schedule(boost::bind(boost::bind(&StoreController::insertTask, this, _1), task));
				break;
			case task::Select:
				this->_selectThreadPool.schedule(boost::bind(boost::bind(&StoreController::selectTask, this, _1), task));
				break;
			case task::Flush:
				this->_insertThreadPool.wait();
				this->_insertThreadPool.schedule(boost::bind(boost::bind(&StoreController::flushTask, this, _1), task));
				this->_insertThreadPool.wait();
				break;
			case task::Info:
				this->_selectThreadPool.schedule(boost::bind(boost::bind(&StoreController::infoTask, this, _1), task));
				break;
			case task::Error:
				LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT("Got task with type ERROR"));
		}
	}

	boost::container::vector<ullintPair>* StoreController::getDataLocationInfo(Query* query)
	{
		boost::container::vector<ullintPair>* dataLocationInfo =
				new boost::container::vector<ullintPair>();
		boost::container::vector<ullintPair>* locations = nullptr;

		// if query should be filtered ask StoreBuffer for data location on GPU
		if(query->metrics.size())
		{
			BOOST_FOREACH(metric_type &m, query->metrics)
			{
				if(_buffers->count(m))	// if elements with such a metric exist in store
				{
					locations = (*_buffers)[m]->Select(query->timePeriods);
					dataLocationInfo->insert(dataLocationInfo->end(), locations->begin(), locations->end());
					delete locations;
					locations = nullptr;
				}
			}
		}
		else // all dataLocationInfos should be returned
		{
			for(auto it=_buffers->begin(); it!=_buffers->end(); it++)
			{
				locations = it->second->Select(query->timePeriods);
				dataLocationInfo->insert(dataLocationInfo->end(), locations->begin(), locations->end());
				delete locations;
				locations = nullptr;
			}
		}

		return dataLocationInfo;
	}

	void StoreController::insertTask(task::Task_Pointer task)
	{
		// Check possible errors
		if(task == nullptr || task->GetType() != task::Insert)
		{
			throw std::runtime_error("Error in insertTask function - wrong argument");
		}

		try
		{
			// SET DEVICE TODO: Can be done once per thread
			this->_cudaController->SetProperDevice();

			// GET store element from task data
			storeElement* element = static_cast<storeElement*>(task->GetData());

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
		catch(std::exception& ex)
		{
			LOG4CPLUS_ERROR_FMT(this->_logger, "Insert task error with exception - [%s] [FAILED]", ex.what());
			task->SetResult(false, ex.what(), nullptr, 0);
		}
		catch(...)
		{
			task->SetResult(false, nullptr, nullptr, 0);
			LOG4CPLUS_FATAL(this->_logger, LOG4CPLUS_TEXT("Insert task error [FAILED]"));
		}
	}

	boost::container::vector<ullintPair>* splitVector(boost::container::vector<ullintPair>* v, int partSize, int partNum)
	{
		unsigned int size = v->size();
		unsigned int start = partNum*partSize;
		unsigned int end = (partNum+1)*partSize;
		end = end >= size ? size : end;
		return new boost::container::vector<ullintPair>(v->begin()+start, v->begin()+end);
	}

	void StoreController::selectTask(task::Task_Pointer task)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Select task [BEGIN]"));

		// Check possible errors
		if(task == nullptr || task->GetType() != task::Select)
		{
			LOG4CPLUS_ERROR(this->_logger, LOG4CPLUS_TEXT("select Task function - wrong argument [FAILED]"));
			throw std::runtime_error("Error in select Task function - wrong argument");
		}

		try
		{
			// SET DEVICE TODO: Can be done once per thread
			this->_cudaController->SetProperDevice();

			// Create query from task data
			Query* query = new Query(task->GetData());
			task->SetQuery(query);
			LOG4CPLUS_INFO(this->_logger, "Select task - " << query->toString());

			// Get data locations
			boost::container::vector<ullintPair>* dataLocationInfo = getDataLocationInfo(query);
			void* queryResult = nullptr;
			size_t size = 0;

			// Break the job into smaller ones if necessary
			int jobPartCount = (dataLocationInfo->size() + this->_maxLocationsPerJob - 1) / this->_maxLocationsPerJob;

			if(jobPartCount > 1)	// split dataLocationInfo
			{
				printf("\n\njobPartCount = %d\n\n", jobPartCount);

				task->SetPart(jobPartCount);

				boost::container::vector<ullintPair>* dataLocationInfoPart;
				for(int i=0; i<jobPartCount; i++)
				{
					dataLocationInfoPart = splitVector(dataLocationInfo, this->_maxLocationsPerJob, i);

					// Execute query
					size = this->_queryCore->ExecuteQuery(&queryResult, query, dataLocationInfoPart);

					// Set query part result
					delete dataLocationInfoPart;
					task->SetResult(true, nullptr, queryResult, size);
					delete static_cast<char*>(queryResult);
					queryResult = nullptr;
				}
			}
			else			// Job doesn't have to be split
			{
				if(jobPartCount == 1) // Execute query
					size = this->_queryCore->ExecuteQuery(&queryResult, query, dataLocationInfo);

				// Set query result
				task->SetResult(true, nullptr, queryResult, size);
			}

			delete dataLocationInfo;
			delete static_cast<char*>(queryResult);
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
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Select task [END]"));
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
			// SET DEVICE TODO: Can be done once per thread
			this->_cudaController->SetProperDevice();

			// TODO: REPAIR FLUSH - and make integration tests for it
			for(Buffers_Map::iterator it = _buffers->begin(); it != _buffers->end(); ++it)
			{
				it->second->Flush();
			}

			task->SetResult(true, "", nullptr, 0);
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
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Info task [BEGIN]"));
		try
		{
			storeNodeInfo* queryResult = new storeNodeInfo();
			// SET DEVICE TODO: Can be done once per thread
			this->_cudaController->SetProperDevice();

			size_t sizeOfResult = this->_infoCore->GetNodeInfo(&queryResult);
			task->SetResult(true, nullptr, (void*)queryResult, sizeOfResult);
			LOG4CPLUS_DEBUG(this->_logger, queryResult->toString());
			delete queryResult;
		}
		catch (std::exception& ex)
		{
			LOG4CPLUS_ERROR_FMT(this->_logger,
					"Info task error with exception - [%s] [FAILED]", ex.what());
			task->SetResult(false, ex.what(), nullptr, 0);
		}
		catch (...)
		{
			task->SetResult(false, nullptr, nullptr, 0);
			LOG4CPLUS_FATAL(this->_logger,
					LOG4CPLUS_TEXT("Info task error [FAILED]"));
		}
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Info task [END]"));
	}

} /* namespace store */
} /* namespace ddj */
