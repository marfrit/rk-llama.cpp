// Concurrent multi-context rknn_matmul probe for RK3588.
// A single context rejects a multi-core mask, so 3-core throughput comes from
// running one single-core context per NPU core and submitting concurrently.
// This measures the vendor library's true multi-core int8 GEMM ceiling.
//
// Build:  gcc -O2 -I/usr/include/rknn -o mm_concurrent mm_concurrent.c -lrknnrt -lpthread
// Run:    stop any other NPU process first (single-tenant), then ./mm_concurrent
//
// Findings (librknnrt 2.3.2, RK3588 @ 1.0 GHz, 2026-07-10):
//   1 context  0.66 TOPS   (33% of 1 core)
//   2 contexts 1.26 TOPS   (1.9x)
//   3 contexts 1.83 TOPS   (2.77x, 30.5% of the rated 6.0)
// => 6 TOPS is a convolution figure; the GEMM ceiling is ~1.83 TOPS, silicon-limited.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
static int M=4096,K=4096,N=4096,ITERS=20;
typedef struct { int core; int done; } arg_t;

static void* worker(void*p){
    arg_t*a=(arg_t*)p;
    rknn_matmul_info info; memset(&info,0,sizeof(info));
    info.M=M; info.K=K; info.N=N; info.type=RKNN_INT8_MM_INT8_TO_INT32;
    info.B_layout=1; info.B_quant_type=0; info.AC_layout=0;
    rknn_matmul_io_attr io; memset(&io,0,sizeof(io));
    rknn_matmul_ctx ctx=0;
    if(rknn_matmul_create(&ctx,&info,&io)!=0){ a->done=-1; return 0; }
    rknn_matmul_set_core_mask(ctx,a->core);        // 1,2,4 = core 0,1,2
    rknn_tensor_mem*A=rknn_create_mem(ctx,io.A.size);
    rknn_tensor_mem*B=rknn_create_mem(ctx,io.B.size);
    rknn_tensor_mem*C=rknn_create_mem(ctx,io.C.size);
    memset(A->virt_addr,1,io.A.size); memset(B->virt_addr,1,io.B.size);
    rknn_matmul_set_io_mem(ctx,A,&io.A); rknn_matmul_set_io_mem(ctx,B,&io.B); rknn_matmul_set_io_mem(ctx,C,&io.C);
    for(int i=0;i<ITERS;i++) rknn_matmul_run(ctx);
    a->done=1; return 0;
}

static double run_n(int ncores){
    pthread_t th[3]; arg_t ar[3];
    int masks[3]={1,2,4};
    double t0=now();
    for(int i=0;i<ncores;i++){ ar[i].core=masks[i]; ar[i].done=0; pthread_create(&th[i],0,worker,&ar[i]); }
    for(int i=0;i<ncores;i++) pthread_join(th[i],0);
    double dt=now()-t0;
    return 2.0*(double)M*K*N*ITERS*ncores/dt/1e12;
}

int main(void){
    printf("=== concurrent single-core contexts (mask 1/2/4), int8 4096^3 ===\n");
    for(int nc=1;nc<=3;nc++){
        double tops=run_n(nc);
        printf("  %d context(s): %5.2f TOPS aggregate  (%4.1f%% of 6.0, %4.1f%% of %d-core rated)\n",
               nc, tops, 100.0*tops/6.0, 100.0*tops/(2.0*nc), nc);
    }
    return 0;
}
