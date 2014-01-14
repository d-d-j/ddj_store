#include "CompressionTest.h"

namespace ddj{
namespace store{

	TEST_F(CompressionTest, Compress_And_Decompress_5000_StoreElements)
	{
		const int N = 5000;
		size_t dataSize = N*sizeof(storeElement);
		//PREPARE
		storeElement host_data[N];
		for(int i=0; i<N; i++)
		{
			host_data[i].time = 21474830000 + i;
			host_data[i].tag = i%256;
			host_data[i].metric = (i*7)%256;
			host_data[i].value = sin(i);
		}
		storeElement* device_data;
		CUDA_CHECK_RETURN( cudaMalloc(&device_data, dataSize) );
		CUDA_CHECK_RETURN( cudaMemcpy(device_data, host_data, dataSize, cudaMemcpyHostToDevice) );

		//TEST
		void* compressedData;
		size_t compressed_size = _compression.CompressTrunk(device_data, dataSize, &compressedData);
		storeElement* decompressedData;
		size_t decompressed_size = _compression.DecompressTrunk(compressedData, compressed_size, &decompressedData);

		//CHECK
		ASSERT_EQ(dataSize, decompressed_size);

		storeElement host_decompressedData[N];
		CUDA_CHECK_RETURN( cudaMemcpy(&host_decompressedData, decompressedData, decompressed_size, cudaMemcpyDeviceToHost) );
		for(int j=0; j<N; j++)
		{
			EXPECT_EQ(host_data[j].time, host_decompressedData[j].time);
			EXPECT_EQ(host_data[j].tag, host_decompressedData[j].tag);
			EXPECT_EQ(host_data[j].metric, host_decompressedData[j].metric);
			EXPECT_FLOAT_EQ(host_data[j].value, host_decompressedData[j].value);
		}

		//CLEAN
		CUDA_CHECK_RETURN( cudaFree(device_data) );
		CUDA_CHECK_RETURN( cudaFree(compressedData) );
		CUDA_CHECK_RETURN( cudaFree(decompressedData) );
	}

}
}
