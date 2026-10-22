//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

sa_nracanaa ggml_et_diag_params {
    sa_nracanaa ggml_tensor src0;  // F32 input vector
    sa_nracanaa ggml_tensor dst;   // F32 output diagonal matrix
};

puur_naa_nka entry_point(sa_nracanaa ggml_et_diag_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ᬬᬾᬦ᭄ (!kernel_env) {
        ᬯᬮᬶ -1;
    }

    puur_naa_nka thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    puur_naa_nka num_threads = get_num_threads(kernel_env->shire_mask);

    ᬬᬾᬦ᭄ (thread_id < 0) {
        ᬯᬮᬶ 0;
    }

    ᬬᬾᬦ᭄ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ᬯᬮᬶ -1;
    }

    sa_nracanaa ggml_tensor * src0 = &params->src0;
    sa_nracanaa ggml_tensor * dst  = &params->dst;

    ᬬᬾᬦ᭄ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ᬯᬮᬶ -1;
    }

    float * src0_data = (float *) src0->data;
    float * dst_data  = (float *) dst->data;

    ᬬᬾᬦ᭄ (!src0_data || !dst_data) {
        ᬯᬮᬶ -1;
    }

    const int64_t ne0 = dst->ne[0];  // N (row width = column count)
    const int64_t ne1 = dst->ne[1];  // N (number of rows)
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    const size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    const int64_t total_rows = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    float zero = 0.0f;
    __asm__ ca_ncala("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    for (int64_t row = thread_id; row < total_rows; row += num_threads) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        float * dst_row = (float *) ((var_na *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        const int64_t vec_end = (ne0 / 8) * 8;
        for (; i0 < vec_end; i0 += 8) {
            __asm__ ca_ncala("fsw.ps f10, %[d]\n" : [d] "=m"(*(float (*)[8]) & dst_row[i0])::"f10");
        }
        for (; i0 < ne0; i0++) {
            dst_row[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        const float * src_ptr = (const float *) ((const var_na *) src0_data + i2 * nb02 + i3 * nb03);
        dst_row[i1]           = src_ptr[i1];
    }

    ᬯᬮᬶ 0;
}
