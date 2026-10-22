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

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_pad_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ vec_copy_f32(ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ * src, int32_t n) {
    int32_t       i       = 0;
    ଅଚଳ int32_t vec_end = (n / 8) * 8;
    ପାଇଁ (; i < vec_end; i += 8) {
        __asm__ ଅସ୍ଥିର(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(ଦଶମିକ (*)[8]) & dst[i])
            : [s] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & src[i])
            : "f10");
    }
    ପାଇଁ (; i < n; i++) {
        dst[i] = src[i];
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_pad_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ ଦଶମିକ * src0_data = (ଅଚଳ ଦଶମିକ *) src0->data;
    ଦଶମିକ *       dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    // Dst dimensions
    ଅଚଳ int64_t ne0 = dst->ne[0];
    ଅଚଳ int64_t ne1 = dst->ne[1];
    ଅଚଳ int64_t ne2 = dst->ne[2];
    ଅଚଳ int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    ଅଚଳ int64_t nb1_src = src0->nb[1];
    ଅଚଳ int64_t nb2_src = src0->nb[2];
    ଅଚଳ int64_t nb3_src = src0->nb[3];

    // Padding values
    ଅଚଳ int32_t lp1 = params->lp[1];
    ଅଚଳ int32_t rp1 = params->rp[1];
    ଅଚଳ int32_t lp2 = params->lp[2];
    ଅଚଳ int32_t rp2 = params->rp[2];
    ଅଚଳ int32_t lp3 = params->lp[3];
    ଅଚଳ int32_t rp3 = params->rp[3];

    ଅଚଳ int64_t total_rows     = ne1 * ne2 * ne3;
    ଅଚଳ int64_t total_elements = ne0 * total_rows;

    ଯଦି (total_elements == 0) {
        ଫେରନ୍ତୁ 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    ଦଶମିକ zero = 0.0f;
    __asm__ ଅସ୍ଥିର("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    ଯଦି (ne0 % 16 == 0) {
        ପାଇଁ (int64_t row = thread_id; row < total_rows; row += ଅଙ୍କ_threads) {
            ଅଚଳ int64_t i3 = row / (ne1 * ne2);
            ଅଚଳ int64_t i2 = (row / ne1) % ne2;
            ଅଚଳ int64_t i1 = row % ne1;

            ଦଶମିକ * dst_row = dst_data + row * ne0;

            ଯଦି (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } ଅନ୍ୟଥା {
                int64_t       i       = 0;
                ଅଚଳ int64_t vec_end = (ne0 / 8) * 8;
                ପାଇଁ (; i < vec_end; i += 8) {
                    __asm__ ଅସ୍ଥିର("fsw.ps f12, %[d]\n" : [d] "=m"(*(ଦଶମିକ (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        ଫେରନ୍ତୁ 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    ଅଚଳ int64_t elems_per_cl = 16;
    ଅଚଳ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ଅଚଳ int64_t ne1_data_end = ne1 - rp1;
    ଅଚଳ int64_t ne2_data_end = ne2 - rp2;
    ଅଚଳ int64_t ne3_data_end = ne3 - rp3;

    ପାଇଁ (int64_t cl = thread_id; cl < total_cl; cl += ଅଙ୍କ_threads) {
        ଅଚଳ int64_t elem_start = cl * elems_per_cl;
        int64_t       elem_end   = elem_start + elems_per_cl;
        ଯଦି (elem_end > total_elements) {
            elem_end = total_elements;
        }

        ପାଇଁ (int64_t idx = elem_start; idx < elem_end; idx++) {
            ଅଚଳ int64_t i0   = idx % ne0;
            ଅଚଳ int64_t rem  = idx / ne0;
            ଅଚଳ int64_t i1   = rem % ne1;
            ଅଚଳ int64_t rem2 = rem / ne1;
            ଅଚଳ int64_t i2   = rem2 % ne2;
            ଅଚଳ int64_t i3   = rem2 / ne2;

            ଯଦି (i1 >= lp1 && i1 < ne1_data_end && i2 >= lp2 && i2 < ne2_data_end && i3 >= lp3 && i3 < ne3_data_end) {
                ଅଚଳ ଦଶମିକ * sp = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_data[idx]    = *sp;
            } ଅନ୍ୟଥା {
                dst_data[idx] = 0.0f;
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
