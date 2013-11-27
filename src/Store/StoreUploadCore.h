/*
 * GpuUploaderCore.h
 *
 *  Created on: 22-09-2013
 *      Author: ghashd
 */

#ifndef GPUUPLOADCORE_H_
#define GPUUPLOADCORE_H_

namespace ddj
{
namespace store
{

	class StoreUploadCore
	{
		CudaController* _cudaController;
	public:
		StoreUploadCore(CudaController* cudaController);
		virtual ~StoreUploadCore();

		void CopyToGpu(storeElement* hostPointer, storeElement* devicePointer, int numElements, cudaStream_t stream);
		size_t CompressGpuBuffer(storeElement* deviceBufferPointer, int elemToUploadCount, void** result, cudaStream_t stream);
		void AppendToMainStore(void* devicePointer, size_t size, storeTrunkInfo* info);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* GPUUPLOADCORE_H_ */
