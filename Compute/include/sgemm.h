/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

/*
 * Main entry of dense matrix-matrix multiplication kernel
 */

#import <stdlib.h>
#import "hpvm.h"

typedef struct __attribute__((__packed__)) {
    float *A;
    size_t bytesA;
    int lda;
    float *B;
    size_t bytesB;
    int ldb;
    float *C;
    size_t bytesC;
    int ldc;
    int k;
    float alpha;
    float beta;
    int block_x;
    int block_y;
    int grid_x;
    int grid_y;
} SGEMMRootIn;
