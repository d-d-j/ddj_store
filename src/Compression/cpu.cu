#include "cpu.h"


int bitLen(int a)
{
    int l = 1;
    while( (a = a>>1) ) l ++;
    return l;
}

void (*compress_select(int level))(int *,char*) {
    /*printf(" >> %d\n",level);*/
    switch (level) {
        case 2:  return compress2;
        case 3:  return compress3;
        case 4:  return compress4;
        case 5:  return compress5;
        case 6:  return compress6;
        case 7:  return compress7;
        case 8:  return compress8;
        case 9:  return compress9;
        case 10: return compress10;
        case 11: return compress11;
        case 12: return compress12;
        case 13: return compress13;
        case 14: return compress14;
        case 15: return compress15;
        case 16: return compress16;
        default: printf("%d\n",level);assert(0);//TODO: inne przypadki
    }
    return NULL;
}

void (*decompress_select(int level))(char *, int*) {
    /*printf(" >> %d\n",level);*/
    switch (level) {
        case 2:  return  decompress2;
        case 3:  return  decompress3;
        case 4:  return  decompress4;
        case 5:  return  decompress5;
        case 6:  return  decompress6;
        case 7:  return  decompress7;
        case 8:  return  decompress8;
        case 9:  return  decompress9;
        case 10: return decompress10;
        case 11: return decompress11;
        case 12: return decompress12;
        case 13: return decompress13;
        case 14: return decompress14;
        case 15: return decompress15;
        case 16: return decompress16;
        default: assert(!level && 0);//TODO: inne przypadki
    }
    return NULL;
}


char *compress_array(int level, int *in, int len)
{
    void (*f_in)(int *, char *) = compress_select(level);
    char *out;
    int i;
    //TODO: na razie sprawdzamy ze len dzieli sie przez 8; potem usunac
    assert(!(len%8));

    out = (char *)malloc( len/8 * level * sizeof(char));

    for (i = 0; i < len / 8; i++) {
        f_in(in + i * 8,  out + i * level);
    }
    return out;
}

int *decompress_array(int level, char *out, int len)
{
    void (*f_out)(char *, int *) = decompress_select(level);
    int *in;
    int i;

    //TODO: na razie sprawdzamy ze len dzieli sie przez 8; potem usunac
    assert(!(len%8));

    in = (int *)malloc( len * sizeof(int));

    for (i = 0; i < len / 8; i++) {
        f_out(out + i * level,  in + i * 8);
    }
    return in;
}
