#ifndef macros
#define macros 0

// Te makra powinny dzialac prawidlowo nie zaleznie od implementacji

#define SGN(a) (int)((unsigned int)((int)a) >> (sizeof(int) * CHAR_BIT - 1))
#define NBITSTOMASK(n) ((1<<(n)) - 1)
#define GETNBITS(a,n)  ((a) & NBITSTOMASK(n))
#define GETNSGNBITS(a,n,b) ((SGN(a) << (n-1)) | GETNBITS(((a)>>(b-n)), (n-1))) 
#define GETNPBITS(a, n, p) GETNBITS((a>>p), (n)) 
#define RECONSTRUCT(a1, n1, p1, a2, n2, p2) GETNPBITS(a1, n1, p1) << (n2) | GETNPBITS(a2, n2, p2)

#define fillto8(c) (((c + 8 - 1) / 8) * 8)
#define fillto(b,c) (((c + b - 1) / b) * b)
#define fillto4(c) (((c + 4 - 1) / 4) * 4)

#define _unused(x) x __attribute__((unused))
#define convert_struct(n, s)  struct sgn {signed int x:n;} __attribute__((unused)) s

#endif
