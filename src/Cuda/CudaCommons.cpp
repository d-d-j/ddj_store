/*
 * CudaCommon.cpp
 *
 *  Created on: 19-11-2013
 *      Author: ghash
 */

#include "CudaCommons.h"

namespace ddj {
namespace store {
	int CudaCommons::CudaGetDevicesCount()
	{
		int count = 0;
		cudaError_t error = cudaGetDeviceCount(&count);
		if(cudaSuccess != error)
			LOG4CPLUS_ERROR_FMT(this->_logger, "CUDA ERROR - (in CudaCommons::CudaGetDevicesCount()) - %s", cudaGetErrorString(error));
		return count;
	}

	int CudaCommons::CudaGetDevicesCountAndPrint()
	{
		int nDevices = this->CudaGetDevicesCount();
		cudaDeviceProp prop;
		int driverVersion = 0, runtimeVersion = 0;

		LOG4CPLUS_INFO_FMT(this->_logger, "CUDA : Found %d CUDA devices: \n\n", nDevices);
		for (int i = 0; i < nDevices; i++)
		{
			cudaSetDevice(i);
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);
			cudaGetDeviceProperties(&prop, i);

			// Log device query
			LOG4CPLUS_INFO_FMT(this->_logger, "	Device Number: %d\n", i);
			LOG4CPLUS_INFO_FMT(this->_logger, "	Device name: %s\n", prop.name);
			LOG4CPLUS_INFO_FMT(this->_logger, "	CUDA Capability Major/Minor version number:    %d.%d\n", prop.major, prop.minor);
			LOG4CPLUS_INFO_FMT(this->_logger, "	CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
					driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
			LOG4CPLUS_INFO_FMT(this->_logger, "	Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
			LOG4CPLUS_INFO_FMT(this->_logger, "	Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
			LOG4CPLUS_INFO_FMT(this->_logger, "	Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		}
		return nDevices;
	}

	bool CudaCommons::CudaCheckDeviceForRequirements(int n)
	{
		int driverVersion = 0, runtimeVersion = 0;
		cudaDeviceProp prop;
		cudaSetDevice(n);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		cudaGetDeviceProperties(&prop, n);

		if(prop.major < _config->GetIntValue("MIN_CUDA_MAJOR_VER")) return false;
		if(prop.minor < _config->GetIntValue("MIN_CUDA_MINOR_VER")) return false;

		return true;
	}

	cudaError_t CudaCommons::CudaAllocateArray(size_t size, void** array)
	{
		size_t mbSize = this->_config->GetIntValue("MB_SIZE_IN_BYTES");
		size_t freeMemory, totalMemory;
		cudaMemGetInfo(&freeMemory, &totalMemory);

		LOG4CPLUS_INFO_FMT(this->_logger, "CUDA INFO - free memory => %f MB, total memory => %f MB", (float)freeMemory/mbSize, (float)totalMemory/mbSize);

		cudaError_t result = cudaSuccess;
		if(totalMemory <= size)
		{
			result = cudaErrorMemoryAllocation;
			return result;
		}
		result = cudaMalloc((void**)array, size);

		cudaMemGetInfo(&freeMemory, &totalMemory);
		LOG4CPLUS_INFO_FMT(this->_logger, "CUDA INFO - free memory => %f MB, total memory => %f MB", (float)freeMemory/mbSize, (float)totalMemory/mbSize);
		return result;
	}

	void CudaCommons::CudaFreeMemory(void* devPtr)
	{
		size_t freeMemory, totalMemory;
		cudaError_t error = cudaFree(devPtr);
		if(cudaSuccess != error)
			LOG4CPLUS_ERROR_FMT(this->_logger, "CUDA ERROR - (in CudaCommons::CudaFreeMemory) - %s", cudaGetErrorString(error));
		cudaMemGetInfo(&freeMemory, &totalMemory);
	}

	void CudaCommons:: GetMemoryCount(size_t* freeMemory, size_t* totalMemory){
		cudaMemGetInfo(freeMemory, totalMemory);
	}

} /* namespace store */
} /* namespace ddj */
