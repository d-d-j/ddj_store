#ifndef CUDACOMPRESSION_H_
#define CUDACOMPRESSION_H_

#include "CudaIncludes.h"
#include "../Core/Logger.h"
#include "../Core/Config.h"
#include "../Compression/gpu.h"


extern "C"
{
void compressVar(int max_size, int bl, int *dev_data, char *dev_out);

void decompressVar(int max_size, int bl, int *dev_data, char* dev_out);
}

#endif
