/*
 * CudaController.cpp
 *
 *  Created on: 30-10-2013
 *      Author: ghash
 */

#include "CudaController.h"

namespace ddj {
namespace store {

	CudaController::CudaController(int uploadStreamsNum, int queryStreamsNum)
	{
		h_LogThreadDebug("Cuda controller constructor started");

		this->_numUploadStreams = uploadStreamsNum;
		this->_numQueryStreams = queryStreamsNum;
		_uploadStreams = new cudaStream_t[this->_numUploadStreams + 1];
		_queryStreams = new cudaStream_t[this->_numQueryStreams];
		_mainMemoryOffset = 0;

		// ALLOCATE MAIN STORAGE ON GPU
		int i = 1;
		while(gpuAllocateMainArray(MAIN_STORE_SIZE / i, &(this->_mainMemoryPointer)) != cudaSuccess)
			if(i <= GPU_MEMORY_ALLOC_ATTEMPTS) i++;
			else throw std::runtime_error("Cannot allocate main GPU memory in storeController");

		h_LogThreadDebug("Cuda controller constructor ended");
	}

	CudaController::~CudaController()
	{
		// RELEASE UPLOAD STREAMS
		for(int i = 0; i < this->_numUploadStreams; i++)
			cudaStreamDestroy(this->_uploadStreams[i]);
		delete [] this->_uploadStreams;

		// RELEASE QUERY STREAMS
		for(int i = 0; i < this->_numQueryStreams; i++)
				cudaStreamDestroy(this->_queryStreams[i]);
			delete [] this->_queryStreams;

		// RELEASE MAIN GPU STORE MEMORY
		gpuFreeMemory(this->_mainMemoryPointer);
	}

	ullint CudaController::GetMainMemoryOffset()
	{
		boost::mutex::scoped_lock lock(_mutex);
		return this->_mainMemoryOffset;
	}

	void CudaController::SetMainMemoryOffset(ullint offset)
	{
		boost::mutex::scoped_lock lock(_mutex);
		this->_mainMemoryOffset = offset;
	}

	cudaStream_t* CudaController::GetUploadStream(int num)
	{
		return this->_uploadStreams[num];
	}

	cudaStream_t* CudaController::GetQueryStream(int num)
	{
		return this->_queryStreams[num];
	}

	void* CudaController::GetMainMemoryPointer()
	{
		return this->_mainMemoryPointer + this->_mainMemoryOffset;
	}

} /* namespace store */
} /* namespace ddj */
