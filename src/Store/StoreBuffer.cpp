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
	h_LogThreadWithTagDebug("StoreBuffer constructor started", tag);

	this->_tag = tag;
	this->_areBuffersSwitched = false;
	this->_bufferElementsCount = 0;
	this->_backBufferElementsCount = 0;
	this->_gpuUploadMonitor = gpuUploadMonitor;
	this->_bufferInfoTreeMonitor = new BTreeMonitor(tag);
	this->_uploaderBarrier = new boost::barrier(2);

	Config* config = Config::GetInstance();

	// ALLOCATE PINNED MEMORY FOR BUFFERS
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&(this -> _buffer), config->GetValue("STORE_BUFFER_SIZE") * sizeof(storeElement)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&(this -> _backBuffer), config->GetValue("STORE_BUFFER_SIZE") * sizeof(storeElement)));

	// START UPLOADER THRAED
	this->_uploaderThread = new boost::thread(boost::bind(&StoreBuffer::uploaderThreadFunction, this));
	this->_uploaderBarrier->wait();

	h_LogThreadWithTagDebug("StoreBuffer constructor ended", tag);
}

StoreBuffer::~StoreBuffer()
{
	h_LogThreadWithTagDebug("StoreBuffer destructor started", this->_tag);

	// STOP UPLOADER THREAD
	{
		boost::mutex::scoped_lock lock(this->_uploaderMutex);
		h_LogThreadWithTagDebug("StoreBuffer locked uploader's mutex", this->_tag);
		this->_uploaderThread->interrupt();
	}
	this->_uploaderThread->join();

	delete this->_bufferInfoTreeMonitor;
	delete this->_uploaderBarrier;
	delete this->_uploaderThread;

	h_LogThreadWithTagDebug("StoreBuffer destructor ended", this->_tag);
}

void StoreBuffer::Insert(storeElement* element)
{
	Config* config = Config::GetInstance();

	h_LogThreadWithTagDebug("Inserting element to buffer", this->_tag);
	this->_buffer[this->_bufferElementsCount] = *element;
	this->_bufferElementsCount++;
	if(_bufferElementsCount == config->GetValue("STORE_BUFFER_SIZE"))
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
	infoElement* elemToInsertToBTree;
	h_LogThreadWithTagDebug("UploaderThread started", this->_tag);
	boost::unique_lock<boost::mutex> lock(this->_uploaderMutex);
	h_LogThreadWithTagDebug("UploaderThread locked his mutex", this->_tag);
	this->_uploaderBarrier->wait();
	try
	{
		while(1)
		{
			h_LogThreadWithTagDebug("UploaderThread waiting", this->_tag);
			this->_uploaderCond.wait(lock);
			h_LogThreadWithTagDebug("UploaderThread doing his job", this->_tag);

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
		h_LogThreadWithTagDebug("UploaderThread ended as interrupted [Success]", this->_tag);
		return;
	}
	catch(...)
	{
		h_LogThreadWithTagDebug("UploaderThread ended with error [Failure]", this->_tag);
	}
}

void StoreBuffer::switchBuffers()
{
	h_LogThreadWithTagDebug("Switching buffers start", this->_tag);
	boost::mutex::scoped_lock lock(this->_uploaderMutex);
	h_LogThreadWithTagDebug("Uploader mutex locked by switchBuffers", this->_tag);
	this->_areBuffersSwitched = true;
	this->_backBufferElementsCount = this->_bufferElementsCount;
	this->_bufferElementsCount = 0;
	this->_buffer.swap(this->_backBuffer);
	this->_uploaderCond.notify_one();
	h_LogThreadWithTagDebug("Switching buffers end", this->_tag);
}

} /* namespace store */
} /* namespace ddj */
