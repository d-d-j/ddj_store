/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

static const int WORK_SIZE = 256;

__host__ __device__ unsigned int bitreverse(unsigned int number)
{
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

struct bitreverse_functor
{
	__host__ __device__ unsigned int operator()(const unsigned int &x)
	{
		return bitreverse(x);
	}
};

int dosth()
{
	thrust::host_vector<unsigned int> idata(WORK_SIZE);
	thrust::host_vector<unsigned int> odata;
	thrust::device_vector<unsigned int> dv;
	int i;

	for (i = 0; i < WORK_SIZE; i++) {
		idata[i] = i;
	}
	dv = idata;

	thrust::transform(dv.begin(), dv.end(), dv.begin(), bitreverse_functor());

	odata = dv;
	for (int i = 0; i < WORK_SIZE; i++) {
		std::cout << "Input value: " << idata[i] << ", output value: "
				<< odata[i] << std::endl;
	}

	return 0;
}
