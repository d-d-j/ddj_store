/*
 * GpuUploaderCore.cpp
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#include "GpuUploadCore.h"


namespace ddj
{
namespace store
{

void GpuUploadCore::copyToGpu(storeElement* hostPointer, storeElement* devicePointer,
		int numElements)
{


		CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)devicePointer, (void*)hostPointer,
				(size_t) numElements * sizeof(storeElement), cudaMemcpyHostToDevice,
				this->uploadStream));


	h_LogThreadDebug("copy to GPU finished");
}

GpuUploadCore::GpuUploadCore()
{
	h_LogThreadDebug("GpuUploadCore no parameter constructor finished");
}

GpuUploadCore::GpuUploadCore(int numUploadStreams)
{

	this->numUploadStreams = numUploadStreams;
	CUDA_CHECK_RETURN(cudaStreamCreate(&this->uploadStream));

	h_LogThreadDebug("GpuUploadCore constructor finished");
}

GpuUploadCore::~GpuUploadCore()
{

}

} /* namespace store */
} /* namespace ddj */
