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

StoreBuffer::StoreBuffer(tag_type tag, GpuUploadMonitor* gpuUploadMonitor)
{
	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store controller [tag=%d] constructor [BEGIN]", tag);

	this->_tag = tag;
	this->_areBuffersSwitched = false;
	this->_bufferElementsCount = 0;
	this->_backBufferElementsCount = 0;
	this->_gpuUploadMonitor = gpuUploadMonitor;
	this->_bufferInfoTreeMonitor = new BTreeMonitor(tag);
	this->_uploaderBarrier = new boost::barrier(2);

	// ALLOCATE PINNED MEMORY FOR BUFFERS
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&(this -> _buffer), STORE_BUFFER_SIZE * sizeof(storeElement)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&(this -> _backBuffer), STORE_BUFFER_SIZE * sizeof(storeElement)));

	// START UPLOADER THRAED
	this->_uploaderThread = new boost::thread(boost::bind(&StoreBuffer::uploaderThreadFunction, this));
	this->_uploaderBarrier->wait();

	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store controller [tag=%d] constructor [END]", tag);
}

StoreBuffer::~StoreBuffer()
{
	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store controller [tag=%d] destructor [BEGIN]", this->_tag);

	// STOP UPLOADER THREAD
	{
		boost::mutex::scoped_lock lock(this->_uploaderMutex);
		this->_uploaderThread->interrupt();
	}
	this->_uploaderThread->join();

	delete this->_bufferInfoTreeMonitor;
	delete this->_uploaderBarrier;
	delete this->_uploaderThread;

	LOG4CPLUS_DEBUG_FMT(this->_logger, "Store controller [tag=%d] destructor [END]", this->_tag);
}

void StoreBuffer::Insert(storeElement* element)
{
	this->_buffer[this->_bufferElementsCount] = *element;
	this->_bufferElementsCount++;
	if(_bufferElementsCount == STORE_BUFFER_SIZE)
	{
		this->switchBuffers();
	}
}

void StoreBuffer::Flush()
{
	boost::mutex::scoped_lock lock(this->_uploaderMutex);
	this->_backBufferElementsCount = this->_bufferElementsCount;
	this->_bufferElementsCount = 0;
	this->_buffer.swap(this->_backBuffer);

	// UPLOAD BUFFER TO GPU
	infoElement* elemToInsertToBTree =
			this->_gpuUploadMonitor->Upload(&(this->_backBuffer), this->_backBufferElementsCount);

	// INSERT INFO ELEMENT TO B+TREE
	this->_bufferInfoTreeMonitor->Insert(elemToInsertToBTree);

	this->_areBuffersSwitched = false;
}

void StoreBuffer::uploaderThreadFunction()
{
	LOG4CPLUS_DEBUG_FMT(this->_logger, "Uploader thread [tag=%d] [BEGIN]", this->_tag);

	infoElement* elemToInsertToBTree;
	boost::unique_lock<boost::mutex> lock(this->_uploaderMutex);
	this->_uploaderBarrier->wait();
	try
	{
		while(1)
		{
			this->_uploaderCond.wait(lock);
			if(this->_areBuffersSwitched)
			{
				// UPLOAD BUFFER TO GPU
				elemToInsertToBTree = this->_gpuUploadMonitor->Upload(
																&(this->_backBuffer),
																this->_backBufferElementsCount);

				// INSERT INFO ELEMENT TO B+TREE
				this->_bufferInfoTreeMonitor->Insert(elemToInsertToBTree);

				// COMMUNICATE THAT BACK BUFFER WAS SUCCESSFULLY UPLOADED
				this->_areBuffersSwitched = false;
			}
		}
	}
	catch(boost::thread_interrupted& ex)
	{
		LOG4CPLUS_DEBUG_FMT(this->_logger, "Uploader thread [tag=%d] [END]", this->_tag);
		return;
	}
	catch(std::exception& ex)
	{
		LOG4CPLUS_ERROR_FMT(this->_logger, "Uploader thread [tag=%d] failed with exception - [%s] [FAILED]", this->_tag, ex.what());
	}
	catch(...)
	{
		LOG4CPLUS_FATAL_FMT(this->_logger, "Uploader thread [tag=%d] error with unknown reason [FAILED]", this->_tag);
		throw;
	}
}

void StoreBuffer::switchBuffers()
{
	boost::mutex::scoped_lock lock(this->_uploaderMutex);

	LOG4CPLUS_DEBUG_FMT(this->_logger, "Switching buffers in store buffer [tag=%d] [BEGIN]", this->_tag);

	this->_areBuffersSwitched = true;
	this->_backBufferElementsCount = this->_bufferElementsCount;
	this->_bufferElementsCount = 0;
	this->_buffer.swap(this->_backBuffer);
	this->_uploaderCond.notify_one();

	LOG4CPLUS_DEBUG_FMT(this->_logger, "Switching buffers in store buffer [tag=%d] [END]", this->_tag);
}

} /* namespace store */
} /* namespace ddj */
