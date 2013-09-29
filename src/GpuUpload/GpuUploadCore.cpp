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
	int streamSize = numElements / this->numUploadStreams;

	for (int i = 0; i < this->numUploadStreams; i++)
	{
		int offset = i * streamSize;

		CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)&devicePointer[offset], (void*)&hostPointer[offset],
				(size_t) streamSize * sizeof(storeElement), cudaMemcpyHostToDevice,
				this->uploadStreams[i]));

	}
	h_LogThreadDebug("copy to GPU finished");
}

GpuUploadCore::GpuUploadCore()
{
}

GpuUploadCore::GpuUploadCore(int numUploadStreams)
{
	this->uploadStreams = new cudaStream_t[numUploadStreams];
	this->numUploadStreams = numUploadStreams;


	for (int i = 0; i < numUploadStreams; i++)
	{
		CUDA_CHECK_RETURN(cudaStreamCreate(&(this->uploadStreams[i])));
	}
	h_LogThreadDebug("GpuUploadCore constructor finished");
}

GpuUploadCore::~GpuUploadCore()
{

}

} /* namespace store */
} /* namespace ddj */
