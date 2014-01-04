#include "gpu.h"
#include <stdio.h>

// celowe rozwinięcie pętli
// id - id bloku do dekompresji
// off o ile przesunac dane
#define FETCH_DATA1(level, data, id, off, dev_out) __fetch(level, data, id, 0, off, dev_out);
#define FETCH_DATA2(level, data, id, off, dev_out) __fetch(level, data, id, 0, off, dev_out); __fetch(level, data, id, 1, off, dev_out);
#define FETCH_DATA3(level, data, id, off, dev_out) __fetch(level, data, id, 0, off, dev_out); __fetch(level, data, id, 1, off, dev_out); __fetch(level, data, id, 2, off , dev_out);
#define FETCH_DATA4(level, data, id, off, dev_out) __fetch(level, data, id, 0, off, dev_out); __fetch(level, data, id, 1, off, dev_out); __fetch(level, data, id, 2, off , dev_out); __fetch(level, data, id, 3, off, dev_out);
#define FETCH_DATA5(level, data, id, off, dev_out) __fetch(level, data, id, 0, off, dev_out); __fetch(level, data, id, 1, off, dev_out); __fetch(level, data, id, 2, off , dev_out); __fetch(level, data, id, 3, off, dev_out); __fetch(level, data, id, 4, off, dev_out);

#define FIX_BLOCK(level, out, in, id) char *out = (char *) in; out += (id * level) % 4  // przesuniecie w bloku pamieci

#define DECOMP( level, id, in, local_out, external_out) decompress##level(in, local_out); copyDecompressed(local_out, external_out);

typedef void (*decompress_func) (int, int, int *, int *, int *, char *);

__device__ decompress_func pfor_func[15] = {
    gpu_decompress2,  gpu_decompress3,  gpu_decompress4,  gpu_decompress5,
    gpu_decompress6,  gpu_decompress7,  gpu_decompress8,  gpu_decompress9,
    gpu_decompress10, gpu_decompress11, gpu_decompress12, gpu_decompress13,
    gpu_decompress14, gpu_decompress15, gpu_decompress16,
};

typedef void (*compress_func) (int *, char*);

__device__ compress_func var_compress_func[15] = {
    compress2,  compress3,  compress4,  compress5,
    compress6,  compress7,  compress8,  compress9,
    compress10, compress11, compress12, compress13,
    compress14, compress15, compress16,
};

__global__ void compress_var (int bl, int *in, char *out)
{
    long id = threadIdx.x + blockIdx.x * blockDim.x;
    int b_in[8];
    char b_out[32];

    for(int i =0; i < 8; i++)
        b_in[i] = *(in + id * 8 + i);

    var_compress_func[bl-2](b_in, b_out);

    for(int i =0; i < bl; i++)
        out[id*bl+i] = b_out[i];
}

__global__ void decompress_var (int bl, int *out, char *dev_out)
{
    long id = threadIdx.x + blockIdx.x * blockDim.x; // id bloku do dekompresji
    int tmp1[8], tmp2[8]; // tymczasowe tablice do dekompresji

    // dekompresja bloku id, uzywa do dekompresji tablic, wyniki skladuje w data + id * 8
    pfor_func[bl-2](id, 0, tmp1, tmp2, tmp2, dev_out);

    for (int i = 0; i < 8; i++) {
        out[id*8+i] = tmp2[i];
    }
}

inline __device__ void __fetch(int level, int *data, int id, int pos, int off, char* dev_out) {
    data[pos] = *((int*)dev_out + off + (id * level) / 4 + pos);
}

//Pobiera dane i dekompresuje n-ty skompresowany blok
__device__ void copyDecompressed(int *in, int *out)
{
    out[0]=in[0];
    out[1]=in[1];
    out[2]=in[2];
    out[3]=in[3];
    out[4]=in[4];
    out[5]=in[5];
    out[6]=in[6];
    out[7]=in[7];
}

__device__ void gpu_decompress2(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA1(2, r_in, id, offset, dev_out);

    FIX_BLOCK(2, tmp_in, r_in, id);

    DECOMP(2, id, tmp_in, r_out, out);
}


__device__ void gpu_decompress3(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA2(3, r_in, id, offset, dev_out);

    FIX_BLOCK(3, tmp_in, r_in, id);

    DECOMP(3, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress4(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA1(4, r_in, id, offset, dev_out);

    char *tmp_in = (char *) r_in;

    DECOMP(4, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress5(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA2(5, r_in, id, offset, dev_out);

    FIX_BLOCK(5, tmp_in, r_in, id);

    DECOMP(5, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress6(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA2(6, r_in, id, offset, dev_out);

    FIX_BLOCK(6, tmp_in, r_in, id);

    DECOMP(6, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress7(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA3(7, r_in, id, offset, dev_out);

    FIX_BLOCK(7, tmp_in, r_in, id);

    DECOMP(7, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress8(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA2(8, r_in, id, offset, dev_out);

    char * tmp_in = (char *) r_in;

    DECOMP(8, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress9(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA3(9, r_in, id, offset, dev_out);

    FIX_BLOCK(9, tmp_in, r_in, id);

    DECOMP(9, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress10(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA3(10, r_in, id, offset, dev_out);

    FIX_BLOCK(10, tmp_in, r_in, id);

    DECOMP(10, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress11(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA4(11, r_in, id, offset, dev_out);

    FIX_BLOCK(11, tmp_in, r_in, id);

    DECOMP(11, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress12(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA3(12, r_in, id, offset, dev_out);

    FIX_BLOCK(12, tmp_in, r_in, id);

    DECOMP(12, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress13(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA4(13, r_in, id, offset, dev_out);

    FIX_BLOCK(13, tmp_in, r_in, id);

    DECOMP(13, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress14(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA4(14, r_in, id, offset, dev_out);

    FIX_BLOCK(14, tmp_in, r_in, id);

    DECOMP(14, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress15(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA5(15, r_in, id, offset, dev_out);

    FIX_BLOCK(15, tmp_in, r_in, id);

    DECOMP(15, id, tmp_in, r_out, out);
}

__device__ void gpu_decompress16(int id, int offset, int *r_in, int *r_out, int *out, char* dev_out)
{
    FETCH_DATA4(16, r_in, id, offset, dev_out);

    char * tmp_in = (char *) r_in;

    DECOMP(16, id, tmp_in, r_out, out);
}
