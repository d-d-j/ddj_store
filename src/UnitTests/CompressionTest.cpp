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

	TEST_F(CompressionTest, AnalizeTrunkData_LinearDataTrunk)
	{
		//PREPARE
		const int N = 1000;
		int min_time = 537;
		int min_tag = 123;
		int min_metric = 7;
		storeElement host_data[N];
		for(int i=0; i<N; i++)
		{
			host_data[i].time = min_time + i;
			host_data[i].tag = min_tag + (i%5);
			host_data[i].metric = min_metric + (i%3);
			host_data[i].value = 3.0f * i;
		}
		storeElement* device_data;
		CUDA_CHECK_RETURN( cudaMalloc(&device_data, sizeof(storeElement)*N) );
		CUDA_CHECK_RETURN( cudaMemcpy(device_data, host_data, sizeof(storeElement)*N, cudaMemcpyHostToDevice) );

		//TEST
		trunkCompressInfo info = AnalizeTrunkData(device_data, N);

		//CHECK
		EXPECT_EQ(min_tag, info.tag_min);
		EXPECT_EQ(min_metric, info.metric_min);
		EXPECT_EQ(min_time, info.time_min);
		EXPECT_EQ(2, info.bytes & 0x0000FFFF);
		EXPECT_EQ(1, (info.bytes & 0xFF000000) >> 24);
		EXPECT_EQ(1, (info.bytes & 0x00FF0000) >> 16);

		//CLEAN
		CUDA_CHECK_RETURN( cudaFree(device_data) );
	}

	TEST_F(CompressionTest, PrepareElementsForCompression_LinearDataTrunk)
	{
		//PREPARE
		const int N = 1000;
		int min_time = 537;
		int min_tag = 123;
		int min_metric = 7;
		storeElement host_data[N];
		for(int i=0; i<N; i++)
		{
			host_data[i].time = min_time + i;
			host_data[i].tag = min_tag + (i%5);
			host_data[i].metric = min_metric + (i%3);
			host_data[i].value = 3.0f * i;
		}
		storeElement* device_data;
		CUDA_CHECK_RETURN( cudaMalloc(&device_data, sizeof(storeElement)*N) );
		CUDA_CHECK_RETURN( cudaMemcpy(device_data, host_data, sizeof(storeElement)*N, cudaMemcpyHostToDevice) );
		trunkCompressInfo info;
		info.tag_min = min_tag;
		info.metric_min = min_metric;
		info.time_min = min_time;

		//TEST
		PrepareElementsForCompression(device_data, N, info);

		//CHECK
		storeElement host_result[N];
		CUDA_CHECK_RETURN( cudaMemcpy(&host_result, device_data, sizeof(storeElement)*N, cudaMemcpyDeviceToHost) );
		for(int j=0; j<N; j++)
		{
			EXPECT_EQ(j, host_result[j].time);
			EXPECT_EQ(j%5, host_result[j].tag);
			EXPECT_EQ(j%3, host_result[j].metric);
		}

		//CLEAN
		CUDA_CHECK_RETURN( cudaFree(device_data) );
	}

	TEST_F(CompressionTest, EncodeInt32UsingNBytes_ForSmallInt)
	{
		int32_t input = 0x24;
		unsigned char output[2];
		EncodeInt32UsingNBytes(output, input, 2);
		int *actual = (int*)output;
		EXPECT_EQ(input, *actual);
	}

	TEST_F(CompressionTest, EncodeInt32UsingNBytes_ForSmallInt_With_Overflow)
	{
		int32_t input = 0xCAFFE24, expected = 0x24;
		unsigned char output[1];
		EncodeInt32UsingNBytes(output, input, 1);
		EXPECT_EQ(expected, output[0]);
	}

	TEST_F(CompressionTest, EncodeInt64UsingNBytes_ForSmallInt)
	{
		int64_t input = 0xBA5EBALL;
		unsigned char output[2];
		EncodeInt64UsingNBytes(output, input, 6);
		int64_t *actual = (int64_t*)output;
		EXPECT_EQ(input, *actual);
	}

	TEST_F(CompressionTest, EncodeInt64UsingNBytes_ForBigInt)
	{
		int64_t input = 0xDEADBEEF;
		unsigned char output[8];
		EncodeInt64UsingNBytes(output, input, 8);
		int64_t *actual = (int64_t*)output;
		EXPECT_EQ(input, *actual);
	}

	TEST_F(CompressionTest, EncodeInt64UsingNBytes_ForBigInt_With_Overflow)
	{
		int64_t input = 0xDEADBEEF , expected = 0xBEEF;
		unsigned char output[2];
		EncodeInt64UsingNBytes(output, input, 2);
		int64_t *actual = (int64_t*)output;
		EXPECT_EQ(expected, *actual);
	}
}
}
