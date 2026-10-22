//******************************************************************************
// Bare Metal PAD F32 Kernel
// Zero-pads an F32 tensor along dimensions 1-3.
//
// Constraints:
//   - No dim0 padding (lp[0]==0, rp[0]==0)
//   - dst contiguous
//   - src nb[0] == 4 (dim0 contiguous for vectorized reads)
//   - Zero-pad only (no circular mode)
//
// Two paths:
//   Aligned (ne0 % 16 == 0): rows distributed across harts, vectorized.
//   Small   (16 % ne0 == 0): cache-line distributed, scalar per-element.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_pad_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ vec_copy_f32(ಭಿನ್ನ * dst, ಧ್ರುವ ಭಿನ್ನ * src, int32_t n) {
    int32_t       i       = 0;
    ಧ್ರುವ int32_t vec_end = (n / 8) * 8;
    ಪ್ರತಿ (; i < vec_end; i += 8) {
        __asm__ ಚಞ್ಚಲ(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(ಭಿನ್ನ (*)[8]) & dst[i])
            : [s] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src[i])
            : "f10");
    }
    ಪ್ರತಿ (; i < n; i++) {
        dst[i] = src[i];
    }
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_pad_params * params, ಶೂನ್ಯ * env) {
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

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ ಭಿನ್ನ * src0_data = (ಧ್ರುವ ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ *       dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    // Dst dimensions
    ಧ್ರುವ int64_t ne0 = dst->ne[0];
    ಧ್ರುವ int64_t ne1 = dst->ne[1];
    ಧ್ರುವ int64_t ne2 = dst->ne[2];
    ಧ್ರುವ int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    ಧ್ರುವ int64_t nb1_src = src0->nb[1];
    ಧ್ರುವ int64_t nb2_src = src0->nb[2];
    ಧ್ರುವ int64_t nb3_src = src0->nb[3];

    // Padding values
    ಧ್ರುವ int32_t lp1 = params->lp[1];
    ಧ್ರುವ int32_t rp1 = params->rp[1];
    ಧ್ರುವ int32_t lp2 = params->lp[2];
    ಧ್ರುವ int32_t rp2 = params->rp[2];
    ಧ್ರುವ int32_t lp3 = params->lp[3];
    ಧ್ರುವ int32_t rp3 = params->rp[3];

    ಧ್ರುವ int64_t total_rows     = ne1 * ne2 * ne3;
    ಧ್ರುವ int64_t total_elements = ne0 * total_rows;

    ಯದಿ (total_elements == 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    ಭಿನ್ನ zero = 0.0f;
    __asm__ ಚಞ್ಚಲ("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    ಯದಿ (ne0 % 16 == 0) {
        ಪ್ರತಿ (int64_t row = thread_id; row < total_rows; row += ಅಂಕೆ_threads) {
            ಧ್ರುವ int64_t i3 = row / (ne1 * ne2);
            ಧ್ರುವ int64_t i2 = (row / ne1) % ne2;
            ಧ್ರುವ int64_t i1 = row % ne1;

            ಭಿನ್ನ * dst_row = dst_data + row * ne0;

            ಯದಿ (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } ಅನ್ಯಥಾ {
                int64_t       i       = 0;
                ಧ್ರುವ int64_t vec_end = (ne0 / 8) * 8;
                ಪ್ರತಿ (; i < vec_end; i += 8) {
                    __asm__ ಚಞ್ಚಲ("fsw.ps f12, %[d]\n" : [d] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    ಧ್ರುವ int64_t elems_per_cl = 16;
    ಧ್ರುವ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ಧ್ರುವ int64_t ne1_data_end = ne1 - rp1;
    ಧ್ರುವ int64_t ne2_data_end = ne2 - rp2;
    ಧ್ರುವ int64_t ne3_data_end = ne3 - rp3;

    ಪ್ರತಿ (int64_t cl = thread_id; cl < total_cl; cl += ಅಂಕೆ_threads) {
        ಧ್ರುವ int64_t elem_start = cl * elems_per_cl;
        int64_t       elem_end   = elem_start + elems_per_cl;
        ಯದಿ (elem_end > total_elements) {
            elem_end = total_elements;
        }

        ಪ್ರತಿ (int64_t idx = elem_start; idx < elem_end; idx++) {
            ಧ್ರುವ int64_t i0   = idx % ne0;
            ಧ್ರುವ int64_t rem  = idx / ne0;
            ಧ್ರುವ int64_t i1   = rem % ne1;
            ಧ್ರುವ int64_t rem2 = rem / ne1;
            ಧ್ರುವ int64_t i2   = rem2 % ne2;
            ಧ್ರುವ int64_t i3   = rem2 / ne2;

            ಯದಿ (i1 >= lp1 && i1 < ne1_data_end && i2 >= lp2 && i2 < ne2_data_end && i3 >= lp3 && i3 < ne3_data_end) {
                ಧ್ರುವ ಭಿನ್ನ * sp = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_data[idx]    = *sp;
            } ಅನ್ಯಥಾ {
                dst_data[idx] = 0.0f;
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
