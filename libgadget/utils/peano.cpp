#include "peano.h"
#ifdef __BMI2__
#include <immintrin.h>
#endif

/*  The following rewrite of the original function
 *  peano_hilbert_key_old() has been written by MARTIN REINECKE and Claude Sonnet 4.6
 *  It is about a factor 2.3 - 2.5  faster than Volker's old routine!
 *  Claude is faster again by a factor of 1.5
 *  BMI2 (enabled with Haswell+ and -march=native) is faster by a factor 2 again.
 */
static constexpr unsigned char rottable3[48][8] = {
    {36, 28, 25, 27, 10, 10, 25, 27},
    {29, 11, 24, 24, 37, 11, 26, 26},
    {8, 8, 25, 27, 30, 38, 25, 27},
    {9, 39, 24, 24, 9, 31, 26, 26},
    {40, 24, 44, 32, 40, 6, 44, 6},
    {25, 7, 33, 7, 41, 41, 45, 45},
    {4, 42, 4, 46, 26, 42, 34, 46},
    {43, 43, 47, 47, 5, 27, 5, 35},
    {33, 35, 36, 28, 33, 35, 2, 2},
    {32, 32, 29, 3, 34, 34, 37, 3},
    {33, 35, 0, 0, 33, 35, 30, 38},
    {32, 32, 1, 39, 34, 34, 1, 31},
    {24, 42, 32, 46, 14, 42, 14, 46},
    {43, 43, 47, 47, 25, 15, 33, 15},
    {40, 12, 44, 12, 40, 26, 44, 34},
    {13, 27, 13, 35, 41, 41, 45, 45},
    {28, 41, 28, 22, 38, 43, 38, 22},
    {42, 40, 23, 23, 29, 39, 29, 39},
    {41, 36, 20, 36, 43, 30, 20, 30},
    {37, 31, 37, 31, 42, 40, 21, 21},
    {28, 18, 28, 45, 38, 18, 38, 47},
    {19, 19, 46, 44, 29, 39, 29, 39},
    {16, 36, 45, 36, 16, 30, 47, 30},
    {37, 31, 37, 31, 17, 17, 46, 44},
    {12, 4, 1, 3, 34, 34, 1, 3},
    {5, 35, 0, 0, 13, 35, 2, 2},
    {32, 32, 1, 3, 6, 14, 1, 3},
    {33, 15, 0, 0, 33, 7, 2, 2},
    {16, 0, 20, 8, 16, 30, 20, 30},
    {1, 31, 9, 31, 17, 17, 21, 21},
    {28, 18, 28, 22, 2, 18, 10, 22},
    {19, 19, 23, 23, 29, 3, 29, 11},
    {9, 11, 12, 4, 9, 11, 26, 26},
    {8, 8, 5, 27, 10, 10, 13, 27},
    {9, 11, 24, 24, 9, 11, 6, 14},
    {8, 8, 25, 15, 10, 10, 25, 7},
    {0, 18, 8, 22, 38, 18, 38, 22},
    {19, 19, 23, 23, 1, 39, 9, 39},
    {16, 36, 20, 36, 16, 2, 20, 10},
    {37, 3, 37, 11, 17, 17, 21, 21},
    {4, 17, 4, 46, 14, 19, 14, 46},
    {18, 16, 47, 47, 5, 15, 5, 15},
    {17, 12, 44, 12, 19, 6, 44, 6},
    {13, 7, 13, 7, 18, 16, 45, 45},
    {4, 42, 4, 21, 14, 42, 14, 23},
    {43, 43, 22, 20, 5, 15, 5, 15},
    {40, 12, 21, 12, 40, 6, 23, 6},
    {13, 7, 13, 7, 41, 41, 22, 20}
};

static constexpr unsigned char subpix3[48][8] = {
    {0, 7, 1, 6, 3, 4, 2, 5},
    {7, 4, 6, 5, 0, 3, 1, 2},
    {4, 3, 5, 2, 7, 0, 6, 1},
    {3, 0, 2, 1, 4, 7, 5, 6},
    {1, 0, 6, 7, 2, 3, 5, 4},
    {0, 3, 7, 4, 1, 2, 6, 5},
    {3, 2, 4, 5, 0, 1, 7, 6},
    {2, 1, 5, 6, 3, 0, 4, 7},
    {6, 1, 7, 0, 5, 2, 4, 3},
    {1, 2, 0, 3, 6, 5, 7, 4},
    {2, 5, 3, 4, 1, 6, 0, 7},
    {5, 6, 4, 7, 2, 1, 3, 0},
    {7, 6, 0, 1, 4, 5, 3, 2},
    {6, 5, 1, 2, 7, 4, 0, 3},
    {5, 4, 2, 3, 6, 7, 1, 0},
    {4, 7, 3, 0, 5, 6, 2, 1},
    {6, 7, 5, 4, 1, 0, 2, 3},
    {7, 0, 4, 3, 6, 1, 5, 2},
    {0, 1, 3, 2, 7, 6, 4, 5},
    {1, 6, 2, 5, 0, 7, 3, 4},
    {2, 3, 1, 0, 5, 4, 6, 7},
    {3, 4, 0, 7, 2, 5, 1, 6},
    {4, 5, 7, 6, 3, 2, 0, 1},
    {5, 2, 6, 1, 4, 3, 7, 0},
    {7, 0, 6, 1, 4, 3, 5, 2},
    {0, 3, 1, 2, 7, 4, 6, 5},
    {3, 4, 2, 5, 0, 7, 1, 6},
    {4, 7, 5, 6, 3, 0, 2, 1},
    {6, 7, 1, 0, 5, 4, 2, 3},
    {7, 4, 0, 3, 6, 5, 1, 2},
    {4, 5, 3, 2, 7, 6, 0, 1},
    {5, 6, 2, 1, 4, 7, 3, 0},
    {1, 6, 0, 7, 2, 5, 3, 4},
    {6, 5, 7, 4, 1, 2, 0, 3},
    {5, 2, 4, 3, 6, 1, 7, 0},
    {2, 1, 3, 0, 5, 6, 4, 7},
    {0, 1, 7, 6, 3, 2, 4, 5},
    {1, 2, 6, 5, 0, 3, 7, 4},
    {2, 3, 5, 4, 1, 0, 6, 7},
    {3, 0, 4, 7, 2, 1, 5, 6},
    {1, 0, 2, 3, 6, 7, 5, 4},
    {0, 7, 3, 4, 1, 6, 2, 5},
    {7, 6, 4, 5, 0, 1, 3, 2},
    {6, 1, 5, 2, 7, 0, 4, 3},
    {5, 4, 6, 7, 2, 3, 1, 0},
    {4, 3, 7, 0, 5, 2, 6, 1},
    {3, 2, 0, 1, 4, 5, 7, 6},
    {2, 5, 1, 6, 3, 4, 0, 7}
};

/* 2-level merged tables: index is (rotation, (pix_hi<<3)|pix_lo).
 * Each entry encodes the result of two consecutive single-level lookups,
 * halving the number of iterations in the serial dependency chain.
 * Built at compile time so the key function is safe to call from
 * multiple threads without initialisation ordering concerns. */
struct merged_tables {
    unsigned char rot[48][64];
    unsigned char sub[48][64];  /* 6-bit key fragment from two levels */
};

static constexpr struct merged_tables build_merged_tables(void)
{
    struct merged_tables t = {};
    for(int r = 0; r < 48; r++) {
        for(int p = 0; p < 64; p++) {
            const int pix_hi = p >> 3;
            const int pix_lo = p & 7;
            const int r1 = rottable3[r][pix_hi];
            t.rot[r][p] = rottable3[r1][pix_lo];
            t.sub[r][p] = (subpix3[r][pix_hi] << 3) | subpix3[r1][pix_lo];
        }
    }
    return t;
}

static constexpr struct merged_tables merged = build_merged_tables();

/*! This function computes a Peano-Hilbert key for an integer triplet (x,y,z),
 *  with x,y,z in the range between 0 and 2^bits-1.
 */
peano_t peano_hilbert_key(const int x, const int y, const int z, const int bits)
{
    int bit = bits - 1;
    unsigned char rotation = 0;
    peano_t key = 0;

#ifdef __BMI2__
    /* Interleave x,y,z bits into a Morton code: bit 3b+2=x[b], 3b+1=y[b], 3b=z[b] */
    const uint64_t morton = _pdep_u64((unsigned)x, 0x4924924924924924ULL) |
                            _pdep_u64((unsigned)y, 0x2492492492492492ULL) |
                            _pdep_u64((unsigned)z, 0x1249249249249249ULL);
#endif

    /* If bits is odd, handle the MSB alone to keep the remainder even-counted */
    if(bits & 1) {
#ifdef __BMI2__
        const unsigned char pix = (morton >> (3 * bit)) & 7;
#else
        const unsigned char pix = (((x >> bit) & 1) << 2) | (((y >> bit) & 1) << 1) | ((z >> bit) & 1);
#endif
        key      = subpix3[rotation][pix];
        rotation = rottable3[rotation][pix];
        bit--;
    }

    /* Process two levels per iteration */
    for(; bit >= 1; bit -= 2) {
#ifdef __BMI2__
        const unsigned char pix = (morton >> (3 * (bit - 1))) & 63;
#else
        const unsigned char pix_hi = (((x >> bit) & 1) << 2) | (((y >> bit) & 1) << 1) | ((z >> bit) & 1);
        const unsigned char pix_lo = (((x >> (bit-1)) & 1) << 2) | (((y >> (bit-1)) & 1) << 1) | ((z >> (bit-1)) & 1);
        const unsigned char pix    = (pix_hi << 3) | pix_lo;
#endif
        key      = (key << 6) | merged.sub[rotation][pix];
        rotation = merged.rot[rotation][pix];
    }

    return key;
}
