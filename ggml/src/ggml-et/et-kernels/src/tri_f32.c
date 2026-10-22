//******************************************************************************
// Tri F32 Kernel
// Triangular masking: zero out elements outside the triangular region.
//
// tri_type (matches ggml_tri_type enum):
//   0 = UPPER_DIAG:  keep where i0 >= i1
//   1 = UPPER:       keep where i0 >  i1
//   2 = LOWER_DIAG:  keep where i0 <= i1
//   3 = LOWER:       keep where i0 <  i1
//
// Distribution: cache-line aligned chunks of the flat contiguous dst.
// Each element is individually classified as keep or zero based on its
// (i0, i1) coordinates. This avoids cache-line sharing between threads
// when ne0 is not a multiple of 16.
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

sa_nracanaa ggml_et_tri_params {
    sa_nracanaa ggml_tensor src0;
    sa_nracanaa ggml_tensor dst;
    int32_t            tri_type;
};

static anta_hstha puur_naa_nka keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    switch (tri_type) {
        pak_sa TRI_TYPE_LOWER:
            ᬯᬮᬶ i0 < i1;
        pak_sa TRI_TYPE_LOWER_DIAG:
            ᬯᬮᬶ i0 <= i1;
        pak_sa TRI_TYPE_UPPER:
            ᬯᬮᬶ i0 > i1;
        pak_sa TRI_TYPE_UPPER_DIAG:
            ᬯᬮᬶ i0 >= i1;
        default:
            ᬯᬮᬶ 0;
    }
}

puur_naa_nka entry_point(sa_nracanaa ggml_et_tri_params * params, void * env) {
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

    sa_nracanaa ggml_tensor * src0     = &params->src0;
    sa_nracanaa ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    ᬬᬾᬦ᭄ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ᬯᬮᬶ -1;
    }

    float * src0_data = (float *) src0->data;
    float * dst_data  = (float *) dst->data;

    ᬬᬾᬦ᭄ (!src0_data || !dst_data) {
        ᬯᬮᬶ -1;
    }

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    const size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    const int64_t total_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    ᬬᬾᬦ᭄ (ne0 % 16 == 0) {
        float zero = 0.0f;
        __asm__ ca_ncala("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        for (int64_t row = thread_id; row < total_rows; row += num_threads) {
            const int64_t i1 = row % ne1;
            const int64_t i2 = (row / ne1) % ne2;
            const int64_t i3 = row / (ne1 * ne2);

            const float * src_row = (const float *) ((const var_na *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            float *       dst_row = (float *) ((var_na *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            switch (tri_type) {
                pak_sa TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    bha_nga;
                pak_sa TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    bha_nga;
                pak_sa TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    bha_nga;
                pak_sa TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    bha_nga;
                default:
                    ᬯᬮᬶ -1;
            }
            ᬬᬾᬦ᭄ (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            for (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ ca_ncala("fsw.ps f10, %[d]\n" : [d] "=m"(*(float (*)[8]) & dst_row[i0])::"f10");
            }
            for (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            for (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ ca_ncala(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(float (*)[8]) & dst_row[i0])
                    : [s] "m"(*(const float (*)[8]) & src_row[i0])
                    : "f11");
            }
            for (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            for (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ ca_ncala("fsw.ps f10, %[d]\n" : [d] "=m"(*(float (*)[8]) & dst_row[i0])::"f10");
            }
            for (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        ᬯᬮᬶ 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        const int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        const int64_t elems_per_cl   = 16;
        const int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
        const int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        ᬬᬾᬦ᭄ (cl_end > total_cl) {
            cl_end = total_cl;
        }
        ᬬᬾᬦ᭄ (cl_start >= total_cl) {
            ᬯᬮᬶ 0;
        }

        const int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        ᬬᬾᬦ᭄ (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        ᬲᭂᬤᭂᬮ (pos < ee) {
            const int64_t i1 = row_idx % ne1;
            const int64_t i2 = (row_idx / ne1) % ne2;
            const int64_t i3 = row_idx / (ne1 * ne2);

            const float * src_row = (const float *) ((const var_na *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            switch (tri_type) {
                pak_sa TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    bha_nga;
                pak_sa TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    bha_nga;
                pak_sa TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    bha_nga;
                pak_sa TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    bha_nga;
                default:
                    ᬯᬮᬶ -1;
            }
            ᬬᬾᬦ᭄ (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            for (int64_t i0 = col; i0 < end_col; i0++) {
                ᬬᬾᬦ᭄ (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } ᬮᬶᬬ {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    ᬯᬮᬶ 0;
}
