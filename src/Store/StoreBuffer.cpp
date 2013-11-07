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

StoreBuffer::StoreBuffer(metric_type metric, GpuUploadMonitor* gpuUploadMonitor)
{
	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store buffer [metric=%d] constructor [BEGIN]", metric);

	this->_metric = metric;
	this->_areBuffersSwitched = false;
	this->_bufferElementsCount = 0;
	this->_backBufferElementsCount = 0;
	this->_gpuUploadMonitor = gpuUploadMonitor;
	this->_bufferInfoTreeMonitor = new BTreeMonitor(metric);
	this->_uploaderBarrier = new boost::barrier(2);

	// ALLOCATE PINNED MEMORY FOR BUFFERS
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&(this -> _buffer), STORE_BUFFER_SIZE * sizeof(storeElement)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&(this -> _backBuffer), STORE_BUFFER_SIZE * sizeof(storeElement)));

	// START UPLOADER THRAED
	this->_uploaderThread = new boost::thread(boost::bind(&StoreBuffer::uploaderThreadFunction, this));
	this->_uploaderBarrier->wait();

	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store buffer [metric=%d] constructor [END]", metric);
}

StoreBuffer::~StoreBuffer()
{
	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store buffer [metric=%d] destructor [BEGIN]", this->_metric);

	// STOP UPLOADER THREAD
	{
		boost::mutex::scoped_lock lock(this->_uploaderMutex);
		this->_uploaderThread->interrupt();
	}
	this->_uploaderThread->join();

	delete this->_bufferInfoTreeMonitor;
	delete this->_uploaderBarrier;
	delete this->_uploaderThread;

	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store buffer [metric=%d] destructor [END]", this->_metric);
}

void StoreBuffer::Insert(storeElement* element)
{
	boost::mutex::scoped_lock lock(this->_bufferMutex);

	this->_buffer[this->_bufferElementsCount] = *element;
	this->_bufferElementsCount++;
	if(_bufferElementsCount == STORE_BUFFER_SIZE)
	{
		while(this->_areBuffersSwitched)
			this->_bufferCond.wait(lock);
		this->switchBuffers();
	}
}

void StoreBuffer::Flush()
{
	boost::mutex::scoped_lock lock(this->_bufferMutex);

	// Wait for back buffer to be uploaded to GPU
	while(this->_areBuffersSwitched)
			this->_bufferCond.wait(lock);

	// Swap buffers
	this->_backBufferElementsCount = this->_bufferElementsCount;
	this->_bufferElementsCount = 0;
	this->_buffer.swap(this->_backBuffer);

	// UPLOAD BUFFER TO GPU
	infoElement* elemToInsertToBTree =
			this->_gpuUploadMonitor->Upload(&(this->_backBuffer), this->_backBufferElementsCount);

	// INSERT INFO ELEMENT TO B+TREE
	this->_bufferInfoTreeMonitor->Insert(elemToInsertToBTree);
}

void StoreBuffer::uploaderThreadFunction()
{
	LOG4CPLUS_DEBUG_FMT(this->_logger, "Uploader thread [metric=%d] [BEGIN]", this->_metric);

	infoElement* elemToInsertToBTree;
	boost::unique_lock<boost::mutex> lock(this->_uploaderMutex);
	this->_uploaderBarrier->wait();
	try
	{
		while(1)
		{
			this->_uploaderCond.wait(lock);
			{
				LOG4CPLUS_DEBUG_FMT(this->_logger, "Uploader thread is doing his JOB:) [metric=%d] [BEGIN]", this->_metric);

				// UPLOAD BUFFER TO GPU
				elemToInsertToBTree = this->_gpuUploadMonitor->Upload(
																&(this->_backBuffer),
																this->_backBufferElementsCount);

				// INSERT INFO ELEMENT TO B+TREE
				this->_bufferInfoTreeMonitor->Insert(elemToInsertToBTree);

				// COMMUNICATE THAT BACK BUFFER WAS SUCCESSFULLY UPLOADED
				boost::mutex::scoped_lock bufferLock(this->_bufferMutex);
				this->_areBuffersSwitched = false;
				this->_bufferCond.notify_one();

				LOG4CPLUS_DEBUG_FMT(this->_logger, "Uploader thread ended his JOB:) [metric=%d] [END]", this->_metric);
			}
		}
	}
	catch(boost::thread_interrupted& ex)
	{
		LOG4CPLUS_DEBUG_FMT(this->_logger, "Uploader thread [metric=%d] [END]", this->_metric);
		return;
	}
	catch(std::exception& ex)
	{
		LOG4CPLUS_ERROR_FMT(this->_logger, "Uploader thread [metric=%d] failed with exception - [%s] [FAILED]", this->_metric, ex.what());
	}
	catch(...)
	{
		LOG4CPLUS_FATAL_FMT(this->_logger, "Uploader thread [metric=%d] error with unknown reason [FAILED]", this->_metric);
		throw;
	}
}

void StoreBuffer::switchBuffers()
{
	LOG4CPLUS_DEBUG_FMT(this->_logger, "Switching buffers in store buffer [metric=%d] [BEGIN]", this->_metric);

	this->_areBuffersSwitched = true;
	this->_backBufferElementsCount = this->_bufferElementsCount;
	this->_bufferElementsCount = 0;
	this->_buffer.swap(this->_backBuffer);
	this->_uploaderCond.notify_one();

	LOG4CPLUS_DEBUG_FMT(this->_logger, "Switching buffers in store buffer [metric=%d] [END]", this->_metric);
}

} /* namespace store */
} /* namespace ddj */
