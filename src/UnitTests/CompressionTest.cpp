#include "CompressionTest.h"
namespace ddj
{
namespace store
{

void big_random_block( int size, int limit , int *data) {
    for (int i=0; i<size; i++)
        data[i] = rand() % limit;
}

TEST_F(CompressionTest, Compress_And_Decompress_Data)
{

	int max_size = 800000;
	char *dev_out;
	int *dev_data, *dev_data2;
	int *host_data, *host_data2;

	cudaMallocHost((void**) &host_data, max_size * sizeof(int));
	cudaMallocHost((void**) &host_data2, max_size * sizeof(int));

	big_random_block(max_size, 10, host_data);

	cudaMalloc((void **) &dev_out, max_size);

	cudaMalloc((void **) &dev_data, max_size * sizeof(int));
	cudaMalloc((void **) &dev_data2, max_size * sizeof(int));
	cudaMemcpy(dev_data, host_data, max_size * sizeof(int),
			cudaMemcpyHostToDevice);

	compressVar(max_size, 5, dev_data, dev_out);

	decompressVar(max_size, 5, dev_data2, dev_out);

	cudaMemcpy(host_data2, dev_data2, max_size * sizeof(int),
			cudaMemcpyDeviceToHost);
	cudaFree(dev_out);
	cudaFree(dev_data);
	cudaFree(dev_data2);

	for (int i = 0; i < max_size; i++)
	{
		ASSERT_EQ(host_data[i], host_data2[i]);
	}
}

}
}
