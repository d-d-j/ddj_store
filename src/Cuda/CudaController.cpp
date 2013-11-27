#include "CudaController.h"

namespace ddj {
namespace store {

	CudaController::CudaController(int uploadStreamsCount, int queryStreamsCount)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller constructor [BEGIN]"));

		this->_uploadStreamsSemaphore = new Semaphore(queryStreamsCount);
		this->_queryStreamsSemaphore = new Semaphore(uploadStreamsCount);
		this->_uploadStreams = new boost::lockfree::queue<cudaStream_t>(uploadStreamsCount);
		this->_queryStreams = new boost::lockfree::queue<cudaStream_t>(queryStreamsCount);

		int i;
		cudaStream_t stream;

		CUDA_CHECK_RETURN( cudaStreamCreate(&stream) );
		this->_syncStream = stream;

		for(i = 0; i < uploadStreamsCount; i++)
		{
			CUDA_CHECK_RETURN( cudaStreamCreate(&stream) );
			this->_uploadStreams->push(stream);
		}

		for(i=0; i < queryStreamsCount; i++)
		{
			CUDA_CHECK_RETURN( cudaStreamCreate(&stream) );
			this->_queryStreams->push(stream);
		}

		this->_mainMemoryOffset = 0;
		this->_mainMemoryPointer = NULL;

		this->allocateMainGpuStorage();

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller constructor [END]"));
	}

	CudaController::~CudaController()
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller destructor [BEGIN]"));

		cudaStream_t stream;

		// RELEASE UPLOAD STREAMS
		while(this->_uploadStreams->pop(stream))
			CUDA_CHECK_RETURN( cudaStreamDestroy(stream) );
		delete this->_uploadStreams;

		// RELEASE QUERY STREAMS
		while(this->_queryStreams->pop(stream))
			CUDA_CHECK_RETURN( cudaStreamDestroy(stream) );
		delete this->_queryStreams;

		// RELEASE MAIN GPU STORE MEMORY
		_cudaCommons.CudaFreeMemory(this->_mainMemoryPointer);

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller destructor [END]"));
	}

	/* STREAMS */

	cudaStream_t CudaController::GetSyncStream()
	{
		return this->_syncStream;
	}

	cudaStream_t CudaController::GetUploadStream()
	{
		this->_uploadStreamsSemaphore->Wait();
		cudaStream_t stream;
		this->_uploadStreams->pop(stream);
		return stream;
	}

	cudaStream_t CudaController::GetQueryStream()
	{
		this->_queryStreamsSemaphore->Wait();
		cudaStream_t stream;
		this->_queryStreams->pop(stream);
		return stream;
	}

	void CudaController::ReleaseUploadStream(cudaStream_t stream)
	{
		this->_uploadStreams->push(stream);
		this->_uploadStreamsSemaphore->Release();
	}

	void CudaController::ReleaseQueryStream(cudaStream_t stream)
	{
		this->_queryStreams->push(stream);
		this->_queryStreamsSemaphore->Release();
	}

	/* MAIN MEMORY */

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

	void CudaController::allocateMainGpuStorage()
	{
		int maxAttempts = _config->GetIntValue("GPU_MEMORY_ALLOC_ATTEMPTS");
		int memorySize = _config->GetIntValue("MAIN_STORE_SIZE");
		size_t mbSize = this->_config->GetIntValue("MB_SIZE_IN_BYTES");
		cudaError_t error = cudaSuccess;
		while(maxAttempts)
		{
			LOG4CPLUS_INFO(this->_logger, "Allocating " << (float)memorySize/mbSize << " MB of memory...");

			error = _cudaCommons.CudaAllocateArray(memorySize, &(this->_mainMemoryPointer));

			if(error != cudaSuccess)
			{
				LOG4CPLUS_ERROR(this->_logger, "CUDA ERROR - Can't allocate " << (float)memorySize/mbSize << " MB of GPU memory - " << cudaGetErrorString(error));
			}
			else
			{
				LOG4CPLUS_INFO(this->_logger, "CUDA SUCCESS - allocated " << (float)memorySize/mbSize << " MB of GPU memory");
				break;
			}
			maxAttempts--;
			memorySize /= 2;
		}
		if(!maxAttempts)	// if memory cannot be allocated throw an exception
		{
			LOG4CPLUS_FATAL_FMT(this->_logger, "CUDA FATAL ERROR MAIN MEMORY ALLOCATION - %s", cudaGetErrorString(error));
			throw std::runtime_error("Cannot allocate main GPU memory in storeController");
		}
	}

} /* namespace store */
} /* namespace ddj */
