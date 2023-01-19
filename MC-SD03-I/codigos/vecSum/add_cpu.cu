/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void add( int N, float *a, float *b, float *c ) {
    int tid = 0;    // this is CPU zero, so we start at zero
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        tid += 1;   // we have one CPU, so we increment by one
    }
}

int main(int argc, char* argv[])
{
    int N = 1;
    if (argc > 1) N = atoi(argv[1]);
    float *a, *b, *c;
    a = (float *)malloc( N * sizeof (float));
    b = (float *)malloc( N * sizeof (float));
    c = (float *)malloc( N * sizeof (float));

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }
    struct timeval begin, end;
    gettimeofday(&begin, NULL);
    add( N, a, b, c );
    gettimeofday(&end, NULL);

    double cpuTime = 1000000*(double)(end.tv_sec - begin.tv_sec);
    cpuTime +=  (double)(end.tv_usec - begin.tv_usec);

    // display the results
    /*for (int i=0; i<N; i++) {
        printf( "%6.0f + %6.0f = %6.0f\n", a[i], b[i], c[i] );
    }*/

    // print times
    printf("\nExecution Time (microseconds): %9.2f\n", cpuTime);

    // free the memory allocated on the CPU
    free(a); a=NULL;
    free(b); b=NULL;
    free(c); c=NULL;

    return 0;
}
