#ifndef COMPRESSIONTEST_H
#define COMPRESSIONTEST_H

#include <gtest/gtest.h>
#include "../Cuda/CudaCompression.cuh"
#include "../Compression/gpu.h"
#include "../Compression/cpu.h"

namespace ddj {
namespace store {

	class CompressionTest : public testing::Test {
	};

}
}
#endif
