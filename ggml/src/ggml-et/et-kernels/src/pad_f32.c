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

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_pad_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
ساكن مضمن فراغ vec_copy_f32(عائم * dst, ثابت عائم * src, int32_t n) {
    int32_t       i       = 0;
    ثابت int32_t vec_end = (n / 8) * 8;
    لكل (; i < vec_end; i += 8) {
        __asm__ متطاير(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(عائم (*)[8]) & dst[i])
            : [s] "m"(*(ثابت عائم (*)[8]) & src[i])
            : "f10");
    }
    لكل (; i < n; i++) {
        dst[i] = src[i];
    }
}

صحيح entry_point(بنية ggml_et_pad_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح لمب_threads = get_لمب_threads(kernel_env->shire_mask);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت عائم * src0_data = (ثابت عائم *) src0->data;
    عائم *       dst_data  = (عائم *) dst->data;

    إذا (!src0_data || !dst_data) {
        أرجع -1;
    }

    // Dst dimensions
    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    ثابت int64_t nb1_src = src0->nb[1];
    ثابت int64_t nb2_src = src0->nb[2];
    ثابت int64_t nb3_src = src0->nb[3];

    // Padding values
    ثابت int32_t lp1 = params->lp[1];
    ثابت int32_t rp1 = params->rp[1];
    ثابت int32_t lp2 = params->lp[2];
    ثابت int32_t rp2 = params->rp[2];
    ثابت int32_t lp3 = params->lp[3];
    ثابت int32_t rp3 = params->rp[3];

    ثابت int64_t total_rows     = ne1 * ne2 * ne3;
    ثابت int64_t total_elements = ne0 * total_rows;

    إذا (total_elements == 0) {
        أرجع 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    عائم zero = 0.0f;
    __asm__ متطاير("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    إذا (ne0 % 16 == 0) {
        لكل (int64_t row = thread_id; row < total_rows; row += لمب_threads) {
            ثابت int64_t i3 = row / (ne1 * ne2);
            ثابت int64_t i2 = (row / ne1) % ne2;
            ثابت int64_t i1 = row % ne1;

            عائم * dst_row = dst_data + row * ne0;

            إذا (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                ثابت عائم * src_row = (ثابت عائم *) ((ثابت حرف *) src0_data + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } وإلا {
                int64_t       i       = 0;
                ثابت int64_t vec_end = (ne0 / 8) * 8;
                لكل (; i < vec_end; i += 8) {
                    __asm__ متطاير("fsw.ps f12, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        أرجع 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    ثابت int64_t elems_per_cl = 16;
    ثابت int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ثابت int64_t ne1_data_end = ne1 - rp1;
    ثابت int64_t ne2_data_end = ne2 - rp2;
    ثابت int64_t ne3_data_end = ne3 - rp3;

    لكل (int64_t cl = thread_id; cl < total_cl; cl += لمب_threads) {
        ثابت int64_t elem_start = cl * elems_per_cl;
        int64_t       elem_end   = elem_start + elems_per_cl;
        إذا (elem_end > total_elements) {
            elem_end = total_elements;
        }

        لكل (int64_t idx = elem_start; idx < elem_end; idx++) {
            ثابت int64_t i0   = idx % ne0;
            ثابت int64_t rem  = idx / ne0;
            ثابت int64_t i1   = rem % ne1;
            ثابت int64_t rem2 = rem / ne1;
            ثابت int64_t i2   = rem2 % ne2;
            ثابت int64_t i3   = rem2 / ne2;

            إذا (i1 >= lp1 && i1 < ne1_data_end && i2 >= lp2 && i2 < ne2_data_end && i3 >= lp3 && i3 < ne3_data_end) {
                ثابت عائم * sp = (ثابت عائم *) ((ثابت حرف *) src0_data + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_data[idx]    = *sp;
            } وإلا {
                dst_data[idx] = 0.0f;
            }
        }
    }

    أرجع 0;
}
