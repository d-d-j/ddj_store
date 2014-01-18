#ifndef compression
#define compression 0
/* compression.c */
#include <limits.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "base.h"

int bitLen(int a);


void (*compress_select(int level))(int *, char *);
void (*decompress_select(int level))(char *, int *);

char *compress_array(int level, int *in, int len);
int *decompress_array(int level, char *out, int len);
#endif
