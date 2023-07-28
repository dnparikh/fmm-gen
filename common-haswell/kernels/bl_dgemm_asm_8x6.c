/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "bl_dgemm_kernel.h"

#define DGEMM_INPUT_GS_BETA_NZ \
	"vmovlpd    (%%rcx        ),  %%xmm0,  %%xmm0  \n\t" \
	"vmovhpd    (%%rcx,%%rsi,1),  %%xmm0,  %%xmm0  \n\t" \
	"vmovlpd    (%%rcx,%%rsi,2),  %%xmm1,  %%xmm1  \n\t" \
	"vmovhpd    (%%rcx,%%r13  ),  %%xmm1,  %%xmm1  \n\t" \
	"vperm2f128 $0x20,   %%ymm1,  %%ymm0,  %%ymm0  \n\t" /*\
	"vmovlpd    (%%rcx,%%rsi,4),  %%xmm2,  %%xmm2  \n\t" \
	"vmovhpd    (%%rcx,%%r15  ),  %%xmm2,  %%xmm2  \n\t" \
	"vmovlpd    (%%rcx,%%r13,2),  %%xmm1,  %%xmm1  \n\t" \
	"vmovhpd    (%%rcx,%%r10  ),  %%xmm1,  %%xmm1  \n\t" \
	"vperm2f128 $0x20,   %%ymm1,  %%ymm2,  %%ymm2  \n\t"*/

#define DGEMM_OUTPUT_GS_BETA_NZ \
	"vextractf128  $1, %%ymm0,  %%xmm1           \n\t" \
	"vmovlpd           %%xmm0,  (%%rcx        )  \n\t" \
	"vmovhpd           %%xmm0,  (%%rcx,%%rsi  )  \n\t" \
	"vmovlpd           %%xmm1,  (%%rcx,%%rsi,2)  \n\t" \
	"vmovhpd           %%xmm1,  (%%rcx,%%r13  )  \n\t" /*\
	"vextractf128  $1, %%ymm2,  %%xmm1           \n\t" \
	"vmovlpd           %%xmm2,  (%%rcx,%%rsi,4)  \n\t" \
	"vmovhpd           %%xmm2,  (%%rcx,%%r15  )  \n\t" \
	"vmovlpd           %%xmm1,  (%%rcx,%%r13,2)  \n\t" \
	"vmovhpd           %%xmm1,  (%%rcx,%%r10  )  \n\t"*/

void bl_dgemm_asm_8x6
     (
       int               k0,
       double*     a,
       double*     b,
       double*     c,
       unsigned long long ldc,
       aux_t *aux
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	u_int64_t k_iter = (unsigned long long)k0 / 4;
	u_int64_t k_left = (unsigned long long)k0 % 4;
	u_int64_t rs_c   = 1;
	u_int64_t cs_c   = ldc;

    double alpha_val = 1.0;
    double beta_val = 0.0;

    double* restrict alpha = &alpha_val;
    double* restrict beta = &beta_val;

	__asm__ volatile
	(
	"                                            \n\t"
	"vzeroall                                    \n\t" // zero all xmm/ymm registers.
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
    "                                            \n\t"
	"addq           $32 * 4, %%rax               \n\t"
	"                                            \n\t" // initialize loop by pre-loading
	"vmovapd           -4 * 32(%%rax), %%ymm0    \n\t"
	"vmovapd           -3 * 32(%%rax), %%ymm1    \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(double)
	"                                            \n\t"
	"leaq   (%%rdi,%%rdi,2), %%r13               \n\t" // r13 = 3*cs_c;
	"leaq   (%%rcx,%%r13,1), %%rdx               \n\t" // rdx = c + 3*cs_c;
	"prefetcht0   7 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetcht0   7 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*cs_c
	"prefetcht0   7 * 8(%%rcx,%%rdi,2)           \n\t" // prefetch c + 2*cs_c
	"prefetcht0   7 * 8(%%rdx)                   \n\t" // prefetch c + 3*cs_c
	"prefetcht0   7 * 8(%%rdx,%%rdi)             \n\t" // prefetch c + 4*cs_c
	"prefetcht0   7 * 8(%%rdx,%%rdi,2)           \n\t" // prefetch c + 5*cs_c
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
    "addq $4 * 6 * 8, %%r15 \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"prefetcht0   64 * 8(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastsd       0 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd       1 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd       2 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd       3 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd       4 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd       5 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"vmovapd           -2 * 32(%%rax), %%ymm0    \n\t"
	"vmovapd           -1 * 32(%%rax), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vbroadcastsd       6 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd       7 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd       8 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd       9 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd      10 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd      11 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"vmovapd            0 * 32(%%rax), %%ymm0    \n\t"
	"vmovapd            1 * 32(%%rax), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"prefetcht0   76 * 8(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastsd      12 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd      13 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd      14 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd      15 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd      16 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd      17 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"vmovapd            2 * 32(%%rax), %%ymm0    \n\t"
	"vmovapd            3 * 32(%%rax), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vbroadcastsd      18 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd      19 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd      20 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd      21 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd      22 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd      23 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"addq           $4 * 8 * 8, %%rax            \n\t" // a += 4*8 (unroll x mr)
	"addq           $4 * 6 * 8, %%rbx            \n\t" // b += 4*6 (unroll x nr)
	"                                            \n\t"
	"vmovapd           -4 * 32(%%rax), %%ymm0    \n\t"
	"vmovapd           -3 * 32(%%rax), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"prefetcht0   64 * 8(%%rax)                  \n\t"
	"                                            \n\t"
	"vbroadcastsd       0 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd       1 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
	"                                            \n\t"
	"vbroadcastsd       2 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd       3 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
	"                                            \n\t"
	"vbroadcastsd       4 *  8(%%rbx), %%ymm2    \n\t"
	"vbroadcastsd       5 *  8(%%rbx), %%ymm3    \n\t"
	"vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
	"vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
	"vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
	"                                            \n\t"
	"addq           $1 * 8 * 8, %%rax            \n\t" // a += 1*8 (unroll x mr)
	"addq           $1 * 6 * 8, %%rbx            \n\t" // b += 1*6 (unroll x nr)
	"                                            \n\t"
	"vmovapd           -4 * 32(%%rax), %%ymm0    \n\t"
	"vmovapd           -3 * 32(%%rax), %%ymm1    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastsd    (%%rax), %%ymm0             \n\t" // load alpha and duplicate
	"vbroadcastsd    (%%rbx), %%ymm3             \n\t" // load beta and duplicate
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm4   \n\t" // scale by alpha
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm5   \n\t"
	"vmulpd           %%ymm0,  %%ymm6,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm7,  %%ymm7   \n\t"
	"vmulpd           %%ymm0,  %%ymm8,  %%ymm8   \n\t"
	"vmulpd           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
	"vmulpd           %%ymm0,  %%ymm10, %%ymm10  \n\t"
	"vmulpd           %%ymm0,  %%ymm11, %%ymm11  \n\t"
	"vmulpd           %%ymm0,  %%ymm12, %%ymm12  \n\t"
	"vmulpd           %%ymm0,  %%ymm13, %%ymm13  \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm14  \n\t"
	"vmulpd           %%ymm0,  %%ymm15, %%ymm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(double)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c +  4*rs_c;
	"                                            \n\t"
	"leaq   (%%rsi,%%rsi,2), %%r13               \n\t" // r13 = 3*rs_c;
	//"leaq   (%%rsi,%%rsi,4), %%r15               \n\t" // r15 = 5*rs_c;
	//"leaq   (%%r13,%%rsi,4), %%r10               \n\t" // r10 = 7*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	".DDONE:                                     \n\t"
	"                                            \n\t"
	"vzeroupper                                  \n\t"
	"                                            \n\t"

	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c)   // 8,
	//   "m" (a_next)*/  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}
