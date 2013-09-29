/*
 * GpuUploaderCore.h
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#ifndef GPUUPLOADCORE_H_
#define GPUUPLOADCORE_H_

#include "../Store/storeElement.h"
#include "cuda_runtime.h"
#include "../Store/LoggerHelper.h"
#include "../CUDA/cudaIncludes.h"
#include "../CUDA/GpuUpload.cuh"

namespace ddj
{
namespace store
{

class GpuUploadCore
{
	cudaStream_t* uploadStreams;
	int numUploadStreams;

public:

	void copyToGpu(storeElement* hostPointer, storeElement* devicePointer,
			int numElements);

	GpuUploadCore();
	GpuUploadCore(int numUploadStreams);
	virtual ~GpuUploadCore();
};

} /* namespace store */
} /* namespace ddj */
#endif /* GPUUPLOADCORE_H_ */
