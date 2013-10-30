#include "CudaController.h"

namespace ddj {
namespace store {

	CudaController::CudaController(int uploadStreamsNum, int queryStreamsNum)
	{
		this->_numUploadStreams = uploadStreamsNum;
		this->_numQueryStreams = queryStreamsNum;
		_uploadStreams = new cudaStream_t[this->_numUploadStreams];
		_queryStreams = new cudaStream_t[this->_numQueryStreams];

		for(int k=0; k<this->_numUploadStreams; k++)
			cudaStreamCreate(&(_uploadStreams[k]));

		for(int l=0; l<this->_numQueryStreams; l++)
					cudaStreamCreate(&(_queryStreams[l]));

		_mainMemoryOffset = 0;

		// ALLOCATE MAIN STORAGE ON GPU
		int i = 1;
		while(gpuAllocateMainArray(MAIN_STORE_SIZE / i, &(this->_mainMemoryPointer)) != cudaSuccess)
			if(i <= GPU_MEMORY_ALLOC_ATTEMPTS) i++;
			else throw std::runtime_error("Cannot allocate main GPU memory in storeController");
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
		boost::mutex::scoped_lock lock(_offsetMutex);
		return this->_mainMemoryOffset;
	}

	void CudaController::SetMainMemoryOffset(ullint offset)
	{
		boost::mutex::scoped_lock lock(_offsetMutex);
		this->_mainMemoryOffset = offset;
	}

	cudaStream_t CudaController::GetUploadStream(int num)
	{
		return this->_uploadStreams[num];
	}

	cudaStream_t CudaController::GetQueryStream(int num)
	{
		return this->_queryStreams[num];
	}

	void* CudaController::GetMainMemoryPointer()
	{
		boost::mutex::scoped_lock lock(_offsetMutex);
		return (char*)this->_mainMemoryPointer+this->_mainMemoryOffset;
	}

} /* namespace store */
} /* namespace ddj */
