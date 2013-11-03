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

namespace ddj
{
namespace store
{

StoreController::StoreController(int gpuDeviceId)
{
	h_LogThreadDebug("StoreController constructor started");

	this->_gpuDeviceId = gpuDeviceId;
	this->_buffers = new boost::unordered_map<tag_type, StoreBuffer_Pointer>();

	// PREPARE TASK FUNCTIONS DICTIONARY
	this->populateTaskFunctions();

	Config* config = Config::GetInstance();
	// CREATE CUDA CONTROLLER (Controlls gpu store side)
	this->_cudaController = new CudaController(STREAMS_NUM_UPLOAD,
			config->GetIntValue("STREAMS_NUM_QUERY"));

	// CREATE GPU UPLOAD MONITOR
	this->_gpuUploadMonitor = new GpuUploadMonitor(this->_cudaController);

	// CREATE QUERY MONITOR
	this->_queryMonitor = new QueryMonitor(this->_cudaController);

	h_LogThreadDebug("StoreController constructor ended");
}

StoreController::~StoreController()
{
	h_LogThreadDebug("StoreController destructor started");

	delete this->_buffers;
	delete this->_queryMonitor;
	delete this->_gpuUploadMonitor;

	h_LogThreadDebug("StoreController destructor ended");
}

void StoreController::ExecuteTask(StoreTask_Pointer task)
{
	// Fire a function from _TaskFunctions with this taskId
	this->_taskFunctions[task->GetType()](task);
}

void StoreController::populateTaskFunctions()
{
	std::pair<int, taskFunc> pair;

	// INSERT
	pair.first = 1;
	pair.second = boost::bind(&StoreController::insertTaskToDictionary, this,
			_1);
	_taskFunctions.insert(pair);
}

void StoreController::insertTaskToDictionary(StoreTask_Pointer task)
{
	h_LogThreadDebug("Insert task function started");

	// Check possible errors
	if (task == nullptr || task->GetType() != Insert)
	{
		h_LogThreadError("Error in insertTask function - wrong argument");
		throw std::runtime_error(
				"Error in insertTask function - wrong argument");
	}

	// GET store element from task data
	storeElement* element = (storeElement*) (task->GetData());

	// GET buffer with element's tag or create one if not exists
	if (this->_buffers->count(element->tag))	// if such a buffer exists
	{
		(*_buffers)[element->tag]->Insert(element);
	}
	else
	{
		StoreBuffer_Pointer newBuf(
				new StoreBuffer(element->tag, this->_gpuUploadMonitor));
		this->_buffers->insert(
		{ element->tag, newBuf });
	}

	// CHWILOWE WYPISYWANIE WARTOSCI W BAZIE
	size_t s;
	storeElement* els = this->_queryMonitor->GetEverything(s);
	int n = s / sizeof(storeElement);
	if (s > 0)
	{
		printf("\nVALUES IN STORE:\n");
		for (int i = 0; i < n; i++)
			printf("Record[%d] tag:%d series:%d time:%d value:%f\n", i,
					els[i].tag, els[i].series, (int) els[i].time, els[i].value);
		printf("\n");
		if (els != NULL)
			cudaFreeHost(els);
	}

	h_LogThreadDebug("Insert task function ended");
}

} /* namespace store */
} /* namespace ddj */
