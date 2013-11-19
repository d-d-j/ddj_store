#include "CudaController.h"

namespace ddj {
namespace store {

	CudaController::CudaController(int uploadStreamsNum, int queryStreamsNum)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller constructor [BEGIN]"));

		this->_numUploadStreams = uploadStreamsNum;
		this->_numQueryStreams = queryStreamsNum;
		_uploadStreams = new cudaStream_t[this->_numUploadStreams];
		_queryStreams = new cudaStream_t[this->_numQueryStreams];

		for(int k=0; k<this->_numUploadStreams; k++)
			cudaStreamCreate(&(_uploadStreams[k]));

		for(int l=0; l<this->_numQueryStreams; l++)
					cudaStreamCreate(&(_queryStreams[l]));

		this->_mainMemoryOffset = 0;
		this->_mainMemoryPointer = NULL;

		// ALLOCATE MAIN STORAGE ON GPU
		int i = 1;
		while(_cudaCommons.CudaAllocateArray(_config->GetIntValue("MAIN_STORE_SIZE") / i, &(this->_mainMemoryPointer)) != cudaSuccess)
			if(i <= _config->GetIntValue("GPU_MEMORY_ALLOC_ATTEMPTS")) i++;
			else throw std::runtime_error("Cannot allocate main GPU memory in storeController");

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller constructor [END]"));
	}

	CudaController::~CudaController()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller destructor [BEGIN]"));

		// RELEASE UPLOAD STREAMS
		for(int i = 0; i < this->_numUploadStreams; i++)
			cudaStreamDestroy(this->_uploadStreams[i]);
		delete [] this->_uploadStreams;

		// RELEASE QUERY STREAMS
		for(int i = 0; i < this->_numQueryStreams; i++)
				cudaStreamDestroy(this->_queryStreams[i]);
			delete [] this->_queryStreams;

		// RELEASE MAIN GPU STORE MEMORY
		_cudaCommons.CudaFreeMemory(this->_mainMemoryPointer);

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller destructor [END]"));
	}

	ullint CudaController::GetMainMemoryOffset()
	{
		boost::mutex::scoped_lock lock(_offsetMutex);
		return this->_mainMemoryOffset;
	}

	void CudaController::SetMainMemoryOffset(ullint offset)
	{
		boost::mutex::scoped_lock lock(_offsetMutex);
		LOG4CPLUS_DEBUG_FMT(this->_logger, "Setting main memory offset to: %llu", offset);
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
		return this->_mainMemoryPointer;
	}

	void* CudaController::GetMainMemoryFirstFreeAddress()
	{
		boost::mutex::scoped_lock lock(_offsetMutex);
		return (char*)this->_mainMemoryPointer+this->_mainMemoryOffset;
	}

} /* namespace store */
} /* namespace ddj */
