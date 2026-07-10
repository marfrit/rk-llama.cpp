// Clean-room rknn_matmul peak-throughput probe for RK3588.
// Measures the vendor library's int8 GEMM rate directly (no ggml, no attention,
// timing only rknn_matmul_run) to establish the hardware ceiling.
//
// Build:  gcc -O2 -I/usr/include/rknn -o mm_peak mm_peak.c -lrknnrt
// Run:    stop any other NPU process first (single-tenant), then ./mm_peak
//
// Findings (librknnrt 2.3.2, RK3588 @ 1.0 GHz, 2026-07-10):
//   - single-core int8 4096^3 = 0.69 TOPS = 34% of a core's rated 2.0
//   - B_layout (native vs normal) and B_quant_type (per-channel vs per-layer)
//     make NO throughput difference: all 0.69 TOPS.
//   - K > 8192 (the int8 K-limit) in one context is catastrophic (~14x slower);
//     split K host-side instead.
//   - A multi-core mask (RKNN_NPU_CORE_0_1_2 = 7) is REJECTED on a single
//     context and falls back to single-core. Use one context per core and
//     submit concurrently (see mm_concurrent.c).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"
#include "rknn_matmul_api.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static void bench(int M,int K,int N,int b_layout,int b_quant,rknn_core_mask cm,const char*tag){
    rknn_matmul_info info; memset(&info,0,sizeof(info));
    info.M=M; info.K=K; info.N=N;
    info.type=RKNN_INT8_MM_INT8_TO_INT32;
    info.B_layout=b_layout;      // 0 normal, 1 native (perf)
    info.B_quant_type=b_quant;   // 0 per-layer, 1 per-channel
    info.AC_layout=0;
    rknn_matmul_io_attr io; memset(&io,0,sizeof(io));
    rknn_matmul_ctx ctx=0;
    if(rknn_matmul_create(&ctx,&info,&io)!=0){ printf("  %-28s create FAIL\n",tag); return; }
    rknn_matmul_set_core_mask(ctx,cm);
    rknn_tensor_mem*A=rknn_create_mem(ctx,io.A.size);
    rknn_tensor_mem*B=rknn_create_mem(ctx,io.B.size);
    rknn_tensor_mem*C=rknn_create_mem(ctx,io.C.size);
    if(!A||!B||!C){ printf("  %-28s mem FAIL\n",tag); return; }
    memset(A->virt_addr,1,io.A.size); memset(B->virt_addr,1,io.B.size);
    rknn_matmul_set_io_mem(ctx,A,&io.A);
    rknn_matmul_set_io_mem(ctx,B,&io.B);
    rknn_matmul_set_io_mem(ctx,C,&io.C);
    for(int i=0;i<3;i++) rknn_matmul_run(ctx);          // warm
    int iters=30; double t0=now();
    for(int i=0;i<iters;i++) rknn_matmul_run(ctx);
    double dt=now()-t0;
    double tops=2.0*(double)M*K*N*iters/dt/1e12;
    printf("  %-28s M=%-5d K=%-5d N=%-5d  %7.2f ms/run  %5.2f TOPS (%4.1f%% of 6.0)\n",
           tag,M,K,N, dt/iters*1e3, tops, 100.0*tops/6.0);
    rknn_destroy_mem(ctx,A); rknn_destroy_mem(ctx,B); rknn_destroy_mem(ctx,C);
    rknn_matmul_destroy(ctx);
}

int main(void){
    printf("=== rknn_matmul int8 peak (single context; mask 7 falls back to 1 core) ===\n");
    bench(4096,4096,4096,1,1,RKNN_NPU_CORE_0,"native+perchan square");
    bench(512, 4096,4096,1,1,RKNN_NPU_CORE_0,"native+perchan prefill-M512");
    bench(4096,4096,4096,1,0,RKNN_NPU_CORE_0,"native+perlayer square");
    bench(4096,4096,4096,0,0,RKNN_NPU_CORE_0,"normal+perlayer square");
    bench(512, 15360,4096,1,1,RKNN_NPU_CORE_0,"native+perchan K15360 (>limit)");
    return 0;
}
