#include "CudaController.h"

namespace ddj {
namespace store {

	CudaController::CudaController(int uploadStreamsCount, int queryStreamsCount, int cudaDeviceId)
	{
		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller constructor [BEGIN]"));
		this->_cudaDeviceId = cudaDeviceId;
		this->_uploadStreamsSemaphore = new Semaphore(queryStreamsCount);
		this->_queryStreamsSemaphore = new Semaphore(uploadStreamsCount);
		this->_uploadStreams = new boost::lockfree::queue<cudaStream_t>(uploadStreamsCount);
		this->_queryStreams = new boost::lockfree::queue<cudaStream_t>(queryStreamsCount);

		// Use device number _cudaDeviceId
		this->SetProperDevice();

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
		this->_mainMemoryPointer = nullptr;

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
		this->_uploadStreams = nullptr;

		// RELEASE QUERY STREAMS
		while(this->_queryStreams->pop(stream))
			CUDA_CHECK_RETURN( cudaStreamDestroy(stream) );
		delete this->_queryStreams;
		this->_queryStreams = nullptr;

		// RELEASE MAIN GPU STORE MEMORY
		_cudaCommons.CudaFreeMemory(this->_mainMemoryPointer);

		LOG4CPLUS_DEBUG(this->_logger, LOG4CPLUS_TEXT("Cuda controller destructor [END]"));
	}

	int CudaController::GetCudaDeviceId()
	{
		return this->_cudaDeviceId;
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
		// TODO: Implement offset >= _mainMemoryCapacity
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
				LOG4CPLUS_WARN(this->_logger, "CUDA SUCCESS - allocated " << (float)memorySize/mbSize << " MB of GPU memory");
				this->_mainMemoryCapacity = memorySize;
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

	ullint CudaController::GetMainMemoryCapacity()
	{
		return this->_mainMemoryCapacity;
	}

	void CudaController::SetProperDevice()
	{
		int devId = -1;

		CUDA_CHECK_RETURN( cudaGetDevice(&devId) );

		if(devId != this->_cudaDeviceId)
		{
			CUDA_CHECK_RETURN( cudaThreadExit() ); // clears all the runtime state for the current thread
			CUDA_CHECK_RETURN( cudaSetDevice(this->_cudaDeviceId) ); // explicit set the current device for the other calls
		}
	}

} /* namespace store */
} /* namespace ddj */
