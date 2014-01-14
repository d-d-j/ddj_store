#ifndef COMPRESSIONTEST_H
#define COMPRESSIONTEST_H

#include <gtest/gtest.h>
#include "../Compression/Compression.h"
#include "../Store/StoreElement.cuh"
#include "../Cuda/CudaCommons.h"
#include "../Cuda/CudaIncludes.h"

// CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../Core/helper_cuda.h"

namespace ddj {
namespace store {

	class CompressionTest : public testing::Test
	{
	protected:
		CompressionTest()
		{
			CudaCommons cudaC;
			cudaC.SetCudaDeviceWithMaxFreeMem();
		}
		~CompressionTest(){}
		virtual void SetUp()
		{
			const char* argv = "";
			cudaSetDevice(findCudaDevice(0, &argv));
		}

		compression::Compression _compression;
	};

}
}
#endif
