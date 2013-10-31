/*
 * CudaController.h
 *
 *  Created on: 30-10-2013
 *      Author: ghash
 */

#ifndef CUDACONTROLLER_H_
#define CUDACONTROLLER_H_

#include "GpuStore.cuh"
#include "../Store/StoreIncludes.h"

namespace ddj {
namespace store {

	class CudaController : public boost::noncopyable
	{
		cudaStream_t* _uploadStreams;
		int _numUploadStreams;
		cudaStream_t* _queryStreams;
		int _numQueryStreams;
		ullint _mainMemoryOffset;
		boost::mutex _offsetMutex;
		void* _mainMemoryPointer;
	public:
		CudaController(int uploadStreamsNum, int queryStreamsNum);
		virtual ~CudaController();

		cudaStream_t GetUploadStream(int num);
		cudaStream_t GetQueryStream(int num);
		ullint GetMainMemoryOffset();
		void SetMainMemoryOffset(ullint offset);
		void* GetMainMemoryPointer();
		void* GetMainMemoryFirstFreeAddress();
	};

} /* namespace store */
} /* namespace ddj */
#endif /* CUDACONTROLLER_H_ */
