#ifndef COMPRESSION_GPU
#define COMPRESSION_GPU
#include "base.h"


__device__ void gpu_decompress2 (int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress3 (int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress4 (int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress5 (int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress6 (int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress7 (int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress8 (int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress9 (int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress10(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress11(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress12(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress13(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress14(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress15(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);
__device__ void gpu_decompress16(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out);

__global__ void compress_var (int bl, int *in, char *out);
__global__ void decompress_var (int bl, int *out, char* dev_out);
#endif
