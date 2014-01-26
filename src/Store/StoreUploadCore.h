#ifndef GPUUPLOADCORE_H_
#define GPUUPLOADCORE_H_

#include "StoreTrunkInfo.h"
#include "StoreElement.cuh"
#include "../Cuda/CudaController.h"
#include "../Compression/Compression.h"
#include <boost/thread.hpp>

namespace ddj {
namespace store {
	/**
	 *  A class used to save an array of elements to Database.
	 *  A class with one public function Upload which uploads an array of StoreElements to main database memory on GPU.
	 *  It also sorts an array before upload and compress it using Compress class object. It produces a structure
	 *  storeTrunkInfo with information about location of uploaded array in GPU memory.
	 */
	class StoreUploadCore
	{
		CudaController* _cudaController;
		boost::mutex _mutex;
		bool _enableCompression;

		/* LOGGER & CONFIG */
		Logger _logger;
		Config* _config;

	public:
		/**
		* A constructor with CudaController pointer as only parameter.
		* It initializes logger and config and sets a pointer to cuda controller.
		* @param cudaController a pointer to CudaController used to access
		* memory on GPU side used by database, and for getting CUDA streams.
		*/
		StoreUploadCore(CudaController* cudaController);

		/**
		 * A virtual empty destructor.
		 */
		virtual ~StoreUploadCore();

		/**
		 * The only public method used to upload an array of elements (time series records) to main DB's memory on GPU.
		 * @param elements an array of storeElements (time series records stored in DB).
		 * @param elementsToUploadCount a number of elements to upload to GPU.
		 */
		storeTrunkInfo* Upload(storeElement* elements, int elementsToUploadCount);

	private:
		void copyToGpu(storeElement* hostPointer, storeElement* devicePointer, int numElements, cudaStream_t stream);
		size_t compressGpuBuffer(storeElement* deviceBufferPointer, int elemToUploadCount, void** result, cudaStream_t stream);
		void appendToMainStore(void* devicePointer, size_t size, storeTrunkInfo* info);
		storeTrunkInfo* sortTrunkAndPrepareInfo(storeElement* elementsToUpload, int elementsToUploadCount);
	};

} /* namespace store */
} /* namespace ddj */
#endif /* GPUUPLOADCORE_H_ */
