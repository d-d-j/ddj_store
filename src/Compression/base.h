#ifndef COMPRESSION_BASE
#define COMPRESSION_BASE
#include "macros.h"
__device__ __host__ void compress9(int *in, char *out);
__device__ __host__ void decompress9(char *out, int *in);
__device__ __host__ void compress10(int *in, char *out);
__device__ __host__ void decompress10(char *out, int *in);
__device__ __host__ void compress11(int *in, char *out);
__device__ __host__ void decompress11(char *out, int *in);
__device__ __host__ void compress12(int *in, char *out);
__device__ __host__ void decompress12(char *out, int *in);
__device__ __host__ void compress13(int *in, char *out);
__device__ __host__ void decompress13(char *out, int *in);
__device__ __host__ void compress14(int *in, char *out);
__device__ __host__ void decompress14(char *out, int *in);
__device__ __host__ void compress15(int *in, char *out);
__device__ __host__ void decompress15(char *out, int *in);
__device__ __host__ void compress16(int *in, char *out);
__device__ __host__ void decompress16(char *out, int *in);
__device__ __host__ void compress8(int *in, char *out);
__device__ __host__ void decompress8(char *out, int *in);
__device__ __host__ void compress7(int *in, char *out);
__device__ __host__ void decompress7(char *out, int *in);
__device__ __host__ void compress6(int *in, char *out);
__device__ __host__ void decompress6(char *out, int *in);
__device__ __host__ void compress3(int *in, char *out);
__device__ __host__ void decompress3(char *out, int *in);
__device__ __host__ void compress5(int *in, char *out);
__device__ __host__ void decompress5(char *out, int *in);
__device__ __host__ void compress2(int *in, char *out);
__device__ __host__ void decompress2(char *out, int *in);
__device__ __host__ void compress4(int *in, char *out);
__device__ __host__ void decompress4(char *out, int *in);

typedef struct CBLOCK_HEADER {
    char method;
    char level;

    char exc_pos_level;
    char exc_ret_level;

    int len;
    int start;
    // licznik wyjatkow
    int exc_counter;
} CBLOCK_HEADER;

inline int cblock_size(CBLOCK_HEADER h)
{
    //return (fillto8(h.len) * h.level  + fillto8(h.exc_counter) * (h.exc_pos_level + h.exc_ret_level + 1))/8;
    return (fillto(32,h.len) * h.level  + fillto(32,h.exc_counter) * (h.exc_pos_level + h.exc_ret_level + 1))/8;
}

#endif
