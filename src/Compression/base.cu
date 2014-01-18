#include "base.h"

__device__ __host__ void compress2(int * in, char *out)
{
    out[0] =  GETNSGNBITS(in[0], 2, 2) | GETNSGNBITS(in[1], 2, 2) << 2 | GETNSGNBITS(in[2], 2, 2) << 4  | GETNSGNBITS(in[3], 2, 2) << 6;
    out[1] =  GETNSGNBITS(in[4], 2, 2) | GETNSGNBITS(in[5], 2, 2) << 2 | GETNSGNBITS(in[6], 2, 2) << 4  | GETNSGNBITS(in[7], 2, 2) << 6;
}

__device__ __host__ void decompress2(char *out, int * in)
{
    convert_struct(2,s);

    in[0] = s.x = GETNBITS(out[0], 2);
    in[1] = s.x = GETNPBITS(out[0], 2, 2);
    in[2] = s.x = GETNPBITS(out[0], 2, 4);
    in[3] = s.x = GETNPBITS(out[0], 2, 6);

    in[4] = s.x = GETNBITS (out[1], 2);
    in[5] = s.x = GETNPBITS(out[1], 2, 2);
    in[6] = s.x = GETNPBITS(out[1], 2, 4);
    in[7] = s.x = GETNPBITS(out[1], 2, 6);
}

__device__ __host__ void compress3(int * in, char *out)
{
    // 3 3 2
    out[0] = GETNSGNBITS(in[0], 3, 3) | GETNSGNBITS(in[1], 3, 3) << 3| GETNSGNBITS(in[2], 2, 3) << 6;
    // 1 3 3 1
    out[1] = 
        GETNBITS(in[2], 1) | GETNSGNBITS(in[3], 3, 3) << 1| 
        GETNSGNBITS(in[4], 3, 3) << 4 | GETNSGNBITS(in[5], 1, 3) << 7;
    //2 3 3
    out[2] = GETNBITS(in[5], 2) | GETNSGNBITS(in[6], 3, 3) << 2| GETNSGNBITS(in[7], 3, 3) << 5;
}

__device__ __host__ void decompress3(char *out, int * in)
{
    convert_struct(3,s);

    // 3 3 2+1 3 3 1+2 3 3
    in[0] = s.x = GETNBITS(out[0],3); 
    in[1] = s.x = GETNPBITS(out[0],3, 3); 

    in[2] = s.x = RECONSTRUCT(out[0], 2, 6, out[1], 1, 0);
    in[3] = s.x = GETNPBITS(out[1], 3, 1); 
    in[4] = s.x = GETNPBITS(out[1], 3, 4); 

    in[5] = s.x = RECONSTRUCT(out[1], 1, 7, out[2], 2, 0);
    in[6] = s.x = GETNPBITS(out[2], 3, 2); 
    in[7] = s.x = GETNPBITS(out[2], 3, 5); 
}

__device__ __host__ void compress4(int * in, char *out)
{
    out[0] =  GETNSGNBITS(in[0], 4,4) | GETNSGNBITS(in[1], 4, 4) << 4;
    out[1] =  GETNSGNBITS(in[2], 4,4) | GETNSGNBITS(in[3], 4, 4) << 4;
    out[2] =  GETNSGNBITS(in[4], 4,4) | GETNSGNBITS(in[5], 4, 4) << 4;
    out[3] =  GETNSGNBITS(in[6], 4,4) | GETNSGNBITS(in[7], 4, 4) << 4;
}

__device__ __host__ void decompress4(char *out, int * in)
{
    convert_struct(4,s);

    in[0] = s.x = GETNBITS(out[0], 4);
    in[1] = s.x = GETNPBITS(out[0], 4, 4);

    in[2] = s.x = GETNBITS(out[1], 4);
    in[3] = s.x = GETNPBITS(out[1], 4, 4);

    in[4] = s.x = GETNBITS(out[2], 4);
    in[5] = s.x = GETNPBITS(out[2], 4, 4);

    in[6] = s.x = GETNBITS(out[3], 4);
    in[7] = s.x = GETNPBITS(out[3], 4, 4);
}

__device__ __host__ void compress5(int * in, char *out)
{
    out[0] = GETNSGNBITS(in[0], 5, 5) | GETNSGNBITS(in[1], 3, 5) << 5; // 5 3
    out[1] = GETNBITS(in[1], 2)       | GETNSGNBITS(in[2], 5, 5) << 2 | GETNSGNBITS(in[3], 1, 5) << 7; //2 5 1 
    out[2] = GETNBITS(in[3], 4)       | GETNSGNBITS(in[4], 4, 5) << 4; // 4 4  
    out[3] = GETNBITS(in[4], 1)       | GETNSGNBITS(in[5], 5, 5) << 1 | GETNSGNBITS(in[6], 2, 5) << 6; //1 5 2  
    out[4] = GETNBITS(in[6], 3)       | GETNSGNBITS(in[7], 5, 5) << 3; // 3 5   
}

__device__ __host__ void decompress5(char *out, int * in)
{
    // 5 3+2 5 1+4 4+1 5 2+3 5
    convert_struct(5,s);
    in[0] = s.x = GETNBITS(out[0],5); 
    in[1] = s.x = RECONSTRUCT(out[0], 3, 5, out[1], 2, 0);
    in[2] = s.x = GETNPBITS(out[1], 5, 2); 

    in[3] = s.x = RECONSTRUCT(out[1], 1, 7, out[2], 4, 0);
    in[4] = s.x = RECONSTRUCT(out[2], 4, 4, out[3], 1, 0);

    in[5] = s.x = GETNPBITS(out[3], 5, 1); 
    in[6] = s.x = RECONSTRUCT(out[3], 2, 6,  out[4], 3, 0);
    in[7] = s.x = GETNPBITS(out[4], 5, 3); 
}

__device__ __host__ void compress6(int * in, char *out)
{
    //    6,2 4,4 2,6  6,2 4,4 2,6
    out[0] = GETNSGNBITS(in[0], 6, 6) | GETNSGNBITS(in[1], 2, 6) << 6; 
    out[1] = GETNBITS(in[1], 4)       | GETNSGNBITS(in[2], 4, 6) << 4;
    out[2] = GETNBITS(in[2], 2)       | GETNSGNBITS(in[3], 6, 6) << 2; 

    out[3] = GETNSGNBITS(in[4], 6, 6) | GETNSGNBITS(in[5], 2, 6) << 6; 
    out[4] = GETNBITS(in[5], 4)       | GETNSGNBITS(in[6], 4, 6) << 4;
    out[5] = GETNBITS(in[6], 2)       | GETNSGNBITS(in[7], 6, 6) << 2; 
}

__device__ __host__ void decompress6(char *out, int * in)
{
    //    6,2 4,4 2,6  6,2 4,4 2,6
    convert_struct(6,s);
    in[0] = s.x = GETNBITS(out[0],6); 
    in[1] = s.x = RECONSTRUCT(out[0], 2, 6, out[1], 4, 0);
    in[2] = s.x = RECONSTRUCT(out[1], 4, 4, out[2], 2, 0);
    in[3] = s.x = GETNPBITS(out[2], 6, 2);

    in[4] = s.x = GETNBITS(out[3],6); 
    in[5] = s.x = RECONSTRUCT(out[3], 2, 6, out[4], 4, 0);
    in[6] = s.x = RECONSTRUCT(out[4], 4, 4, out[5], 2, 0);
    in[7] = s.x = GETNPBITS(out[5], 6, 2);
}

__device__ __host__ void compress7(int * in, char *out)
{
    // 7,1 6,2 5,3 4,4 3,5 2,6 1,7 
    out[0] = GETNSGNBITS(in[0], 7, 7) | GETNSGNBITS(in[1], 1, 7) << 7; 
    out[1] = GETNBITS(in[1], 6)       | GETNSGNBITS(in[2], 2, 7) << 6;
    out[2] = GETNBITS(in[2], 5)       | GETNSGNBITS(in[3], 3, 7) << 5; 

    out[3] = GETNBITS(in[3], 4)       | GETNSGNBITS(in[4], 4, 7) << 4; 
    out[4] = GETNBITS(in[4], 3)       | GETNSGNBITS(in[5], 5, 7) << 3;
    out[5] = GETNBITS(in[5], 2)       | GETNSGNBITS(in[6], 6, 7) << 2; 
    out[6] = GETNBITS(in[6], 1)       | GETNSGNBITS(in[7], 7, 7) << 1; 
}

__device__ __host__ void decompress7(char *out, int * in)
{
    // 7,1 6,2 5,3 4,4 3,5 2,6 1,7 
    convert_struct(7,s);
    in[0] = s.x = GETNBITS(out[0],7); 
    in[1] = s.x = RECONSTRUCT(out[0], 1, 7, out[1], 6, 0);
    in[2] = s.x = RECONSTRUCT(out[1], 2, 6, out[2], 5, 0);
    in[3] = s.x = RECONSTRUCT(out[2], 3, 5, out[3], 4, 0);

    in[4] = s.x = RECONSTRUCT(out[3], 4, 4, out[4], 3, 0);
    in[5] = s.x = RECONSTRUCT(out[4], 5, 3, out[5], 2, 0);
    in[6] = s.x = RECONSTRUCT(out[5], 6, 2, out[6], 1, 0);
    in[7] = s.x = GETNPBITS(out[6], 7, 1);
}

__device__ __host__ void compress8(int * in, char *out)
{
    out[0] = GETNSGNBITS(in[0], 8, 8);
    out[1] = GETNSGNBITS(in[1], 8, 8);
    out[2] = GETNSGNBITS(in[2], 8, 8);
    out[3] = GETNSGNBITS(in[3], 8, 8);
    out[4] = GETNSGNBITS(in[4], 8, 8);
    out[5] = GETNSGNBITS(in[5], 8, 8);
    out[6] = GETNSGNBITS(in[6], 8, 8);
    out[7] = GETNSGNBITS(in[7], 8, 8);
}

__device__ __host__ void decompress8(char *out, int * in)
{
    convert_struct(8,s);
    in[0] = s.x = GETNBITS(out[0],8); 
    in[1] = s.x = GETNBITS(out[1],8); 
    in[2] = s.x = GETNBITS(out[2],8); 
    in[3] = s.x = GETNBITS(out[3],8); 
    in[4] = s.x = GETNBITS(out[4],8); 
    in[5] = s.x = GETNBITS(out[5],8); 
    in[6] = s.x = GETNBITS(out[6],8); 
    in[7] = s.x = GETNBITS(out[7],8); 
}

__device__ __host__ void compress9(int *in, char *out)
{
    out[0] = GETNSGNBITS(in[0], 8, 9);
    out[1] =    GETNBITS(in[0], 1) | GETNSGNBITS(in[1], 7, 9) << 1;
    out[2] =    GETNBITS(in[1], 2) | GETNSGNBITS(in[2], 6, 9) << 2;
    out[3] =    GETNBITS(in[2], 3) | GETNSGNBITS(in[3], 5, 9) << 3;
    out[4] =    GETNBITS(in[3], 4) | GETNSGNBITS(in[4], 4, 9) << 4;
    out[5] =    GETNBITS(in[4], 5) | GETNSGNBITS(in[5], 3, 9) << 5;
    out[6] =    GETNBITS(in[5], 6) | GETNSGNBITS(in[6], 2, 9) << 6;
    out[7] =    GETNBITS(in[6], 7) | GETNSGNBITS(in[7], 1, 9) << 7;
    out[8] =    GETNBITS(in[7], 8);
}

__device__ __host__ void decompress9(char *out, int *in)
{
    convert_struct(9,s);
    // [8, 1]
    in[0] = s.x = RECONSTRUCT(out[0], 8, 0, out[1], 1, 0);
    // [7, 2]
    in[1] = s.x = RECONSTRUCT(out[1], 7, 1, out[2], 2, 0);
    // [6, 3]
    in[2] = s.x = RECONSTRUCT(out[2], 6, 2, out[3], 3, 0);
    // [5, 4]
    in[3] = s.x = RECONSTRUCT(out[3], 5, 3, out[4], 4, 0);
    // [4, 5]
    in[4] = s.x = RECONSTRUCT(out[4], 4, 4, out[5], 5, 0);
    // [3, 6]
    in[5] = s.x = RECONSTRUCT(out[5], 3, 5, out[6], 6, 0);
    // [2, 7]
    in[6] = s.x = RECONSTRUCT(out[6], 2, 6, out[7], 7, 0);
    // [1, 8]
    in[7] = s.x = RECONSTRUCT(out[7], 1, 7, out[8], 8, 0);

}

__device__ __host__ void compress10(int * in, char *out)
{

    out[0] = GETNSGNBITS(in[0], 8, 10);
    out[1] =    GETNBITS(in[0], 2) | GETNSGNBITS(in[1], 6, 10) << 2;
    out[2] =    GETNBITS(in[1], 4) | GETNSGNBITS(in[2], 4, 10) << 4;
    out[3] =    GETNBITS(in[2], 6) | GETNSGNBITS(in[3], 2, 10) << 6;
    out[4] =    GETNBITS(in[3], 8);
    out[5] = GETNSGNBITS(in[4], 8, 10);
    out[6] =    GETNBITS(in[4], 2) | GETNSGNBITS(in[5], 6, 10) << 2;
    out[7] =    GETNBITS(in[5], 4) | GETNSGNBITS(in[6], 4, 10) << 4;
    out[8] =    GETNBITS(in[6], 6) | GETNSGNBITS(in[7], 2, 10) << 6;
    out[9] =    GETNBITS(in[7], 8);

}


__device__ __host__ void decompress10(char *out, int *in)
{

    convert_struct(10,s);
    // [8, 2]
    in[0] = s.x = RECONSTRUCT(out[0], 8, 0, out[1], 2, 0);
    // [6, 4]
    in[1] = s.x = RECONSTRUCT(out[1], 6, 2, out[2], 4, 0);
    // [4, 6]
    in[2] = s.x = RECONSTRUCT(out[2], 4, 4, out[3], 6, 0);
    // [2, 8]
    in[3] = s.x = RECONSTRUCT(out[3], 2, 6, out[4], 8, 0);
    // [8, 2]
    in[4] = s.x = RECONSTRUCT(out[5], 8, 0, out[6], 2, 0);
    // [6, 4]
    in[5] = s.x = RECONSTRUCT(out[6], 6, 2, out[7], 4, 0);
    // [4, 6]
    in[6] = s.x = RECONSTRUCT(out[7], 4, 4, out[8], 6, 0);
    // [2, 8]
    in[7] = s.x = RECONSTRUCT(out[8], 2, 6, out[9], 8, 0);

}


__device__ __host__ void compress11(int * in, char *out)
{

    out[0] = GETNSGNBITS(in[0], 8, 11);
    out[1] =    GETNBITS(in[0], 3) | GETNSGNBITS(in[1], 5, 11) << 3;
    out[2] =    GETNBITS(in[1], 6) | GETNSGNBITS(in[2], 2, 11) << 6;
    out[3] =    GETNBITS(in[2] >> 1, 8);
    out[4] =    GETNBITS(in[2], 1) | GETNSGNBITS(in[3], 7, 11) << 1;
    out[5] =    GETNBITS(in[3], 4) | GETNSGNBITS(in[4], 4, 11) << 4;
    out[6] =    GETNBITS(in[4], 7) | GETNSGNBITS(in[5], 1, 11) << 7;
    out[7] =    GETNBITS(in[5] >> 2, 8);
    out[8] =    GETNBITS(in[5], 2) | GETNSGNBITS(in[6], 6, 11) << 2;
    out[9] =    GETNBITS(in[6], 5) | GETNSGNBITS(in[7], 3, 11) << 5;
    out[10] =   GETNBITS(in[7], 8);

}


__device__ __host__ void decompress11(char *out, int *in)
{

    convert_struct(11,s);
    // [8, 3]
    in[0] = s.x = RECONSTRUCT(out[0], 8, 0, out[1], 3, 0);
    // [5, 6]
    in[1] = s.x = RECONSTRUCT(out[1], 5, 3, out[2], 6, 0);
    // [2, 8, 1]
    in[2] = s.x = GETNPBITS(out[2], 2, 6) <<9 | GETNBITS(out[3], 8) << 1 | GETNBITS(out[4], 1);
    // [7, 4]
    in[3] = s.x = RECONSTRUCT(out[4], 7, 1, out[5], 4, 0);
    // [4, 7]
    in[4] = s.x = RECONSTRUCT(out[5], 4, 4, out[6], 7, 0);
    // [1, 8, 2]
    in[5] = s.x = GETNPBITS(out[6], 1, 7) <<10| GETNBITS(out[7], 8) << 2 | GETNBITS(out[8], 2);
    // [6, 5]
    in[6] = s.x = RECONSTRUCT(out[8], 6, 2, out[9], 5, 0);
    // [3, 8]
    in[7] = s.x = RECONSTRUCT(out[9], 3, 5, out[10], 8, 0);

}


__device__ __host__ void compress12(int * in, char *out)
{

    out[0] = GETNSGNBITS(in[0], 8, 12);
    out[1] =    GETNBITS(in[0], 4) | GETNSGNBITS(in[1], 4, 12) << 4;
    out[2] =    GETNBITS(in[1], 8);
    out[3] = GETNSGNBITS(in[2], 8, 12);
    out[4] =    GETNBITS(in[2], 4) | GETNSGNBITS(in[3], 4, 12) << 4;
    out[5] =    GETNBITS(in[3], 8);
    out[6] = GETNSGNBITS(in[4], 8, 12);
    out[7] =    GETNBITS(in[4], 4) | GETNSGNBITS(in[5], 4, 12) << 4;
    out[8] =    GETNBITS(in[5], 8);
    out[9] = GETNSGNBITS(in[6], 8, 12);
    out[10] =   GETNBITS(in[6], 4) | GETNSGNBITS(in[7], 4, 12) << 4;
    out[11] =   GETNBITS(in[7], 8);

}


__device__ __host__ void decompress12(char *out, int *in)
{

    convert_struct(12,s);
    // [8, 4]
    in[0] = s.x = RECONSTRUCT(out[0], 8, 0, out[1], 4, 0);
    // [4, 8]
    in[1] = s.x = RECONSTRUCT(out[1], 4, 4, out[2], 8, 0);
    // [8, 4]
    in[2] = s.x = RECONSTRUCT(out[3], 8, 0, out[4], 4, 0);
    // [4, 8]
    in[3] = s.x = RECONSTRUCT(out[4], 4, 4, out[5], 8, 0);
    // [8, 4]
    in[4] = s.x = RECONSTRUCT(out[6], 8, 0, out[7], 4, 0);
    // [4, 8]
    in[5] = s.x = RECONSTRUCT(out[7], 4, 4, out[8], 8, 0);
    // [8, 4]
    in[6] = s.x = RECONSTRUCT(out[9], 8, 0, out[10], 4, 0);
    // [4, 8]
    in[7] = s.x = RECONSTRUCT(out[10], 4, 4, out[11], 8, 0);

}


__device__ __host__ void compress13(int * in, char *out)
{

    out[0] = GETNSGNBITS(in[0], 8, 13);
    out[1] =    GETNBITS(in[0], 5) | GETNSGNBITS(in[1], 3, 13) << 5;
    out[2] =    GETNBITS(in[1] >> 2, 8);
    out[3] =    GETNBITS(in[1], 2) | GETNSGNBITS(in[2], 6, 13) << 2;
    out[4] =    GETNBITS(in[2], 7) | GETNSGNBITS(in[3], 1, 13) << 7;
    out[5] =    GETNBITS(in[3] >> 4, 8);
    out[6] =    GETNBITS(in[3], 4) | GETNSGNBITS(in[4], 4, 13) << 4;
    out[7] =    GETNBITS(in[4] >> 1, 8);
    out[8] =    GETNBITS(in[4], 1) | GETNSGNBITS(in[5], 7, 13) << 1;
    out[9] =    GETNBITS(in[5], 6) | GETNSGNBITS(in[6], 2, 13) << 6;
    out[10] =   GETNBITS(in[6] >> 3, 8);
    out[11] =   GETNBITS(in[6], 3) | GETNSGNBITS(in[7], 5, 13) << 3;
    out[12] =   GETNBITS(in[7], 8);

}


__device__ __host__ void decompress13(char *out, int *in)
{

    convert_struct(13,s);
    // [8, 5]
    in[0] = s.x = RECONSTRUCT(out[0], 8, 0, out[1], 5, 0);
    // [3, 8, 2]
    in[1] = s.x = GETNPBITS(out[1], 3, 5) << 10 | GETNBITS(out[2], 8) << 2 | GETNBITS(out[3], 2);
    // [6, 7]
    in[2] = s.x = RECONSTRUCT(out[3], 6, 2, out[4], 7, 0);
    // [1, 8, 4]
    in[3] = s.x = GETNPBITS(out[4], 1, 7) << 12 | GETNBITS(out[5], 8) << 4 | GETNBITS(out[6], 4);
    // [4, 8, 1]
    in[4] = s.x = GETNPBITS(out[6], 4, 4) << 9  | GETNBITS(out[7], 8) << 1 | GETNBITS(out[8], 1);
    // [7, 6]
    in[5] = s.x = RECONSTRUCT(out[8], 7, 1, out[9], 6, 0);
    // [2, 8, 3]
    in[6] = s.x = GETNPBITS(out[9], 2, 6) << 11 | GETNBITS(out[10], 8) << 3 | GETNBITS(out[11], 3);
    // [5, 8]
    in[7] = s.x = RECONSTRUCT(out[11], 5, 3, out[12], 8, 0);

}


__device__ __host__ void compress14(int * in, char *out)
{

    out[0] = GETNSGNBITS(in[0], 8, 14);
    out[1] =    GETNBITS(in[0], 6) | GETNSGNBITS(in[1], 2, 14) << 6;
    out[2] =    GETNBITS(in[1] >> 4, 8);
    out[3] =    GETNBITS(in[1], 4) | GETNSGNBITS(in[2], 4, 14) << 4;
    out[4] =    GETNBITS(in[2] >> 2, 8);
    out[5] =    GETNBITS(in[2], 2) | GETNSGNBITS(in[3], 6, 14) << 2;
    out[6] =    GETNBITS(in[3], 8);
    out[7] = GETNSGNBITS(in[4], 8, 14);
    out[8] =    GETNBITS(in[4], 6) | GETNSGNBITS(in[5], 2, 14) << 6;
    out[9] =    GETNBITS(in[5] >> 4, 8);
    out[10] =   GETNBITS(in[5], 4) | GETNSGNBITS(in[6], 4, 14) << 4;
    out[11] =   GETNBITS(in[6] >> 2, 8);
    out[12] =   GETNBITS(in[6], 2) | GETNSGNBITS(in[7], 6, 14) << 2;
    out[13] =   GETNBITS(in[7], 8);

}


__device__ __host__ void decompress14(char *out, int *in)
{

    convert_struct(14,s);
    // [8, 6]
    in[0] = s.x = RECONSTRUCT(out[0], 8, 0, out[1], 6, 0);
    // [2, 8, 4]
    in[1] = s.x = GETNPBITS(out[1], 2, 6) << 12 | GETNBITS(out[2], 8) << 4 | GETNBITS(out[3], 4);
    // [4, 8, 2]
    in[2] = s.x = GETNPBITS(out[3], 4, 4) << 10 | GETNBITS(out[4], 8) << 2 | GETNBITS(out[5], 2);
    // [6, 8]
    in[3] = s.x = RECONSTRUCT(out[5], 6, 2, out[6], 8, 0);
    // [8, 6]
    in[4] = s.x = RECONSTRUCT(out[7], 8, 0, out[8], 6, 0);
    // [2, 8, 4]
    in[5] = s.x = GETNPBITS(out[8], 2, 6) << 12 | GETNBITS(out[9], 8) << 4 | GETNBITS(out[10], 4);
    // [4, 8, 2]
    in[6] = s.x = GETNPBITS(out[10], 4, 4) << 10 | GETNBITS(out[11], 8) << 2 | GETNBITS(out[12], 2);
    // [6, 8]
    in[7] = s.x = RECONSTRUCT(out[12], 6, 2, out[13], 8, 0);

}


__device__ __host__ void compress15(int * in, char *out)
{

    out[0] = GETNSGNBITS(in[0], 8, 15);
    out[1] =    GETNBITS(in[0], 7) | GETNSGNBITS(in[1], 1, 15) << 7;
    out[2] =    GETNBITS(in[1] >> 6, 8);
    out[3] =    GETNBITS(in[1], 6) | GETNSGNBITS(in[2], 2, 15) << 6;
    out[4] =    GETNBITS(in[2] >> 5, 8);
    out[5] =    GETNBITS(in[2], 5) | GETNSGNBITS(in[3], 3, 15) << 5;
    out[6] =    GETNBITS(in[3] >> 4, 8);
    out[7] =    GETNBITS(in[3], 4) | GETNSGNBITS(in[4], 4, 15) << 4;
    out[8] =    GETNBITS(in[4] >> 3, 8);
    out[9] =    GETNBITS(in[4], 3) | GETNSGNBITS(in[5], 5, 15) << 3;
    out[10] =   GETNBITS(in[5] >> 2, 8);
    out[11] =   GETNBITS(in[5], 2) | GETNSGNBITS(in[6], 6, 15) << 2;
    out[12] =   GETNBITS(in[6] >> 1, 8);
    out[13] =   GETNBITS(in[6], 1) | GETNSGNBITS(in[7], 7, 15) << 1;
    out[14] =   GETNBITS(in[7], 8);

}


__device__ __host__ void decompress15(char *out, int *in)
{

    convert_struct(15,s);
    // [8, 7]
    in[0] = s.x = RECONSTRUCT(out[0], 8, 0, out[1], 7, 0);
    // [1, 8, 6]
    in[1] = s.x = GETNPBITS(out[1], 1, 7) << 14 | GETNBITS(out[2], 8) << 6 | GETNBITS(out[3], 6);
    // [2, 8, 5]
    in[2] = s.x = GETNPBITS(out[3], 2, 6) << 13 | GETNBITS(out[4], 8) << 5 | GETNBITS(out[5], 5);
    // [3, 8, 4]
    in[3] = s.x = GETNPBITS(out[5], 3, 5) << 12 | GETNBITS(out[6], 8) << 4 | GETNBITS(out[7], 4);
    // [4, 8, 3]
    in[4] = s.x = GETNPBITS(out[7], 4, 4) << 11 | GETNBITS(out[8], 8) << 3 | GETNBITS(out[9], 3);
    // [5, 8, 2]
    in[5] = s.x = GETNPBITS(out[9], 5, 3) << 10 | GETNBITS(out[10], 8) << 2| GETNBITS(out[11], 2);
    // [6, 8, 1]
    in[6] = s.x = GETNPBITS(out[11], 6, 2) << 9 | GETNBITS(out[12], 8) << 1 | GETNBITS(out[13], 1);
    // [7, 8]
    in[7] = s.x = RECONSTRUCT(out[13], 7, 1, out[14], 8, 0);

}

__device__ __host__ void compress16(int * in, char *out)
{
    out[0]  = GETNSGNBITS(in[0], 8, 16);
    out[1]  = GETNBITS(in[0], 8);

    out[2]  = GETNSGNBITS(in[1], 8, 16);
    out[3]  = GETNBITS(in[1], 8);

    out[4]  = GETNSGNBITS(in[2], 8, 16);
    out[5]  = GETNBITS(in[2], 8);

    out[6]  = GETNSGNBITS(in[3], 8, 16);
    out[7]  = GETNBITS(in[3], 8);

    out[8]  = GETNSGNBITS(in[4], 8, 16);
    out[9]  = GETNBITS(in[4], 8);

    out[10] = GETNSGNBITS(in[5], 8, 16);
    out[11]  = GETNBITS(in[5], 8);

    out[12] = GETNSGNBITS(in[6], 8, 16);
    out[13]  = GETNBITS(in[6], 8);

    out[14] = GETNSGNBITS(in[7], 8, 16);
    out[15]  = GETNBITS(in[7], 8);
}

__device__ __host__ void decompress16(char *out, int * in)
{
    convert_struct(16,s);
    in[0] = s.x = RECONSTRUCT(out[0],  8, 0, out[1],  8, 0);
    in[1] = s.x = RECONSTRUCT(out[2],  8, 0, out[3],  8, 0);
    in[2] = s.x = RECONSTRUCT(out[4],  8, 0, out[5],  8, 0);
    in[3] = s.x = RECONSTRUCT(out[6],  8, 0, out[7],  8, 0);
    in[4] = s.x = RECONSTRUCT(out[8],  8, 0, out[9],  8, 0);
    in[5] = s.x = RECONSTRUCT(out[10], 8, 0, out[11], 8, 0);
    in[6] = s.x = RECONSTRUCT(out[12], 8, 0, out[13], 8, 0);
    in[7] = s.x = RECONSTRUCT(out[14], 8, 0, out[15], 8, 0);
}
