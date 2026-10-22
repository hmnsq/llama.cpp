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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

#define TRI_TYPE_UPPER_DIAG 0
#define TRI_TYPE_UPPER      1
#define TRI_TYPE_LOWER_DIAG 2
#define TRI_TYPE_LOWER      3

ಸಂರಚನಾ ggml_et_tri_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor dst;
    int32_t            tri_type;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    ವಿಕಲ್ಪ (tri_type) {
        ಪಕ್ಷ TRI_TYPE_LOWER:
            ಪ್ರತ್ಯಾಗಮ i0 < i1;
        ಪಕ್ಷ TRI_TYPE_LOWER_DIAG:
            ಪ್ರತ್ಯಾಗಮ i0 <= i1;
        ಪಕ್ಷ TRI_TYPE_UPPER:
            ಪ್ರತ್ಯಾಗಮ i0 > i1;
        ಪಕ್ಷ TRI_TYPE_UPPER_DIAG:
            ಪ್ರತ್ಯಾಗಮ i0 >= i1;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ 0;
    }
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_tri_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0     = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t ne0 = dst->ne[0];
    ಧ್ರುವ int64_t ne1 = dst->ne[1];
    ಧ್ರುವ int64_t ne2 = dst->ne[2];
    ಧ್ರುವ int64_t ne3 = dst->ne[3];

    ಧ್ರುವ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಧ್ರುವ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ಧ್ರುವ int64_t total_rows = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    ಯದಿ (ne0 % 16 == 0) {
        ಭಿನ್ನ zero = 0.0f;
        __asm__ ಚಞ್ಚಲ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ಪ್ರತಿ (int64_t row = thread_id; row < total_rows; row += ಅಂಕೆ_threads) {
            ಧ್ರುವ int64_t i1 = row % ne1;
            ಧ್ರುವ int64_t i2 = (row / ne1) % ne2;
            ಧ್ರುವ int64_t i3 = row / (ne1 * ne2);

            ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ಭಿನ್ನ *       dst_row = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_start, keep_end;
            ವಿಕಲ್ಪ (tri_type) {
                ಪಕ್ಷ TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    ಭಙ್ಗ;
                ಪಕ್ಷ TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    ಭಙ್ಗ;
                ಪಕ್ಷ TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    ಭಙ್ಗ;
                ಪಕ್ಷ TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    ಭಙ್ಗ;
                ಸ್ವಭಾವ:
                    ಪ್ರತ್ಯಾಗಮ -1;
            }
            ಯದಿ (keep_end > ne0) {
                keep_end = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            ಪ್ರತಿ (; i0 + 8 <= keep_start; i0 += 8) {
                __asm__ ಚಞ್ಚಲ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_row[i0])::"f10");
            }
            ಪ್ರತಿ (; i0 < keep_start; i0++) {
                dst_row[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            ಪ್ರತಿ (; i0 + 8 <= keep_end; i0 += 8) {
                __asm__ ಚಞ್ಚಲ(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_row[i0])
                    : [s] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src_row[i0])
                    : "f11");
            }
            ಪ್ರತಿ (; i0 < keep_end; i0++) {
                dst_row[i0] = src_row[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            ಪ್ರತಿ (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ ಚಞ್ಚಲ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_row[i0])::"f10");
            }
            ಪ್ರತಿ (; i0 < ne0; i0++) {
                dst_row[i0] = 0.0f;
            }
        }
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ಧ್ರುವ int64_t total_elements = ne0 * ne1 * ne2 * ne3;
        ಧ್ರುವ int64_t elems_per_cl   = 16;
        ಧ್ರುವ int64_t total_cl       = (total_elements + elems_per_cl - 1) / elems_per_cl;

        ಧ್ರುವ int64_t cl_per_thread = (total_cl + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
        ಧ್ರುವ int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        ಯದಿ (cl_end > total_cl) {
            cl_end = total_cl;
        }
        ಯದಿ (cl_start >= total_cl) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        ಧ್ರುವ int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        ಯದಿ (ee > total_elements) {
            ee = total_elements;
        }

        int64_t row_idx = es / ne0;
        int64_t col     = es % ne0;

        int64_t pos = es;
        ಯಾವತ್ (pos < ee) {
            ಧ್ರುವ int64_t i1 = row_idx % ne1;
            ಧ್ರುವ int64_t i2 = (row_idx / ne1) % ne2;
            ಧ್ರುವ int64_t i3 = row_idx / (ne1 * ne2);

            ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t row_remaining   = ne0 - col;
            int64_t chunk_remaining = ee - pos;
            int64_t n               = row_remaining < chunk_remaining ? row_remaining : chunk_remaining;

            int64_t keep_start, keep_end;
            ವಿಕಲ್ಪ (tri_type) {
                ಪಕ್ಷ TRI_TYPE_LOWER:
                    keep_start = 0;
                    keep_end   = i1;
                    ಭಙ್ಗ;
                ಪಕ್ಷ TRI_TYPE_LOWER_DIAG:
                    keep_start = 0;
                    keep_end   = i1 + 1;
                    ಭಙ್ಗ;
                ಪಕ್ಷ TRI_TYPE_UPPER:
                    keep_start = i1 + 1;
                    keep_end   = ne0;
                    ಭಙ್ಗ;
                ಪಕ್ಷ TRI_TYPE_UPPER_DIAG:
                    keep_start = i1;
                    keep_end   = ne0;
                    ಭಙ್ಗ;
                ಸ್ವಭಾವ:
                    ಪ್ರತ್ಯಾಗಮ -1;
            }
            ಯದಿ (keep_end > ne0) {
                keep_end = ne0;
            }

            int64_t end_col = col + n;
            ಪ್ರತಿ (int64_t i0 = col; i0 < end_col; i0++) {
                ಯದಿ (i0 >= keep_start && i0 < keep_end) {
                    dst_data[pos + (i0 - col)] = src_row[i0];
                } ಅನ್ಯಥಾ {
                    dst_data[pos + (i0 - col)] = 0.0f;
                }
            }

            pos += n;
            col = 0;
            row_idx++;
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
