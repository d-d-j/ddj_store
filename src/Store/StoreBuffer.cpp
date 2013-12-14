/*
 * StoreBuffer.cpp
 *
 *  Created on: Aug 10, 2013
 *      Author: Karol Dzitkowski
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

#include "StoreBuffer.h"

namespace ddj {
namespace store {

StoreBuffer::StoreBuffer(metric_type metric, int bufferCapacity, StoreUploadCore* uploadCore)
{
	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store buffer [metric=%d] constructor [BEGIN]", metric);

	this->_metric = metric;
	this->_bufferElementsCount = 0;
	this->_backBufferElementsCount = 0;
	this->_uploadCore = uploadCore;
	this->_bufferInfoTreeMonitor = new btree::BTreeMonitor(metric);
	this->_bufferCapacity = bufferCapacity;
	this->_bufferSize = bufferCapacity * sizeof(storeElement);

	// ALLOCATE MEMORY FOR BUFFERS
	this->_buffer = new storeElement[bufferCapacity];
	this->_backBuffer = new storeElement[bufferCapacity];

	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store buffer [metric=%d] constructor [END]", metric);
}

StoreBuffer::~StoreBuffer()
{
	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store buffer [metric=%d] destructor [BEGIN]", this->_metric);

	delete this->_bufferInfoTreeMonitor;
	delete this->_uploadCore;

	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store buffer [metric=%d] destructor [END]", this->_metric);
}

void StoreBuffer::Insert(storeElement* element)
{
	this->_bufferMutex.lock();
	this->_buffer[this->_bufferElementsCount] = *element;
	this->_bufferElementsCount++;
	LOG4CPLUS_DEBUG(this->_logger, "buffer elem count = " << this->_bufferElementsCount);
	if(_bufferElementsCount == this->_bufferCapacity)
	{
		this->_backBufferMutex.lock();
		this->switchBuffers();
		this->_bufferMutex.unlock();

		// copy buffer to pinned memory
		storeElement* pinnedMemory = nullptr;
		CUDA_CHECK_RETURN( cudaMallocHost((void**)&(pinnedMemory), this->_bufferSize) );
		CUDA_CHECK_RETURN
		(
			cudaMemcpy(pinnedMemory, this->_backBuffer, this->_bufferCapacity * sizeof(storeElement), cudaMemcpyHostToHost);
		)

		this->_backBufferMutex.unlock();

		// UPLOAD BUFFER TO GPU (releases _backBufferMutex when element is already on GPU
		storeTrunkInfo* elemToInsertToBTree = this->_uploadCore->Upload(pinnedMemory, this->_backBufferElementsCount);
		CUDA_CHECK_RETURN( cudaFreeHost(pinnedMemory) );

		// INSERT INFO ELEMENT TO B+TREE
		LOG4CPLUS_DEBUG(this->_logger, "Insert to BTREE [START]");
		this->_bufferInfoTreeMonitor->Insert(elemToInsertToBTree);
		LOG4CPLUS_DEBUG(this->_logger, "Insert to BTREE [END]");

		delete elemToInsertToBTree;
	} else {
	this->_bufferMutex.unlock();
	}
}

void StoreBuffer::Flush()
{
	this->_bufferMutex.lock();
	this->_backBufferMutex.lock();
	this->switchBuffers();
	this->_bufferMutex.unlock();

	// copy buffer to pinned memory
	storeElement* pinnedMemory;
	CUDA_CHECK_RETURN( cudaMallocHost((void**)&(pinnedMemory), this->_bufferSize) );
	CUDA_CHECK_RETURN
	(
		cudaMemcpy(pinnedMemory, this->_backBuffer, this->_bufferCapacity * sizeof(storeElement), cudaMemcpyHostToHost);
	)

	this->_backBufferMutex.unlock();

	// UPLOAD BUFFER TO GPU (releases _backBufferMutex when element is already on GPU
	storeTrunkInfo* elemToInsertToBTree = this->_uploadCore->Upload(this->_backBuffer, this->_backBufferElementsCount);
	CUDA_CHECK_RETURN( cudaFree(pinnedMemory) );

	// INSERT INFO ELEMENT TO B+TREE
	this->_bufferInfoTreeMonitor->Insert(elemToInsertToBTree);
	delete elemToInsertToBTree;
}

void StoreBuffer::switchBuffers()
{
	this->_backBufferElementsCount = this->_bufferElementsCount;
	this->_bufferElementsCount = 0;
	storeElement* temp;
	temp = this->_buffer;
	this->_buffer = this->_backBuffer;
	this->_backBuffer = temp;
}

} /* namespace store */
} /* namespace ddj */
