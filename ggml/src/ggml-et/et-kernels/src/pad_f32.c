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

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_pad_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
நிலையான உள்வரி வெறுமை vec_copy_f32(தசம * dst, மாறிலி தசம * src, int32_t n) {
    int32_t       i       = 0;
    மாறிலி int32_t vec_end = (n / 8) * 8;
    க்கு (; i < vec_end; i += 8) {
        __asm__ நிலையற்ற(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(தசம (*)[8]) & dst[i])
            : [s] "m"(*(மாறிலி தசம (*)[8]) & src[i])
            : "f10");
    }
    க்கு (; i < n; i++) {
        dst[i] = src[i];
    }
}

முழு entry_point(கட்டமைப்பு ggml_et_pad_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    மாறிலி தசம * src0_data = (மாறிலி தசம *) src0->data;
    தசம *       dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;
    }

    // Dst dimensions
    மாறிலி int64_t ne0 = dst->ne[0];
    மாறிலி int64_t ne1 = dst->ne[1];
    மாறிலி int64_t ne2 = dst->ne[2];
    மாறிலி int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    மாறிலி int64_t nb1_src = src0->nb[1];
    மாறிலி int64_t nb2_src = src0->nb[2];
    மாறிலி int64_t nb3_src = src0->nb[3];

    // Padding values
    மாறிலி int32_t lp1 = params->lp[1];
    மாறிலி int32_t rp1 = params->rp[1];
    மாறிலி int32_t lp2 = params->lp[2];
    மாறிலி int32_t rp2 = params->rp[2];
    மாறிலி int32_t lp3 = params->lp[3];
    மாறிலி int32_t rp3 = params->rp[3];

    மாறிலி int64_t total_rows     = ne1 * ne2 * ne3;
    மாறிலி int64_t total_elements = ne0 * total_rows;

    எனில் (total_elements == 0) {
        திரும்பு 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    தசம zero = 0.0f;
    __asm__ நிலையற்ற("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    எனில் (ne0 % 16 == 0) {
        க்கு (int64_t row = thread_id; row < total_rows; row += இலக்கம்_threads) {
            மாறிலி int64_t i3 = row / (ne1 * ne2);
            மாறிலி int64_t i2 = (row / ne1) % ne2;
            மாறிலி int64_t i1 = row % ne1;

            தசம * dst_row = dst_data + row * ne0;

            எனில் (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                மாறிலி தசம * src_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } இல்லையெனில் {
                int64_t       i       = 0;
                மாறிலி int64_t vec_end = (ne0 / 8) * 8;
                க்கு (; i < vec_end; i += 8) {
                    __asm__ நிலையற்ற("fsw.ps f12, %[d]\n" : [d] "=m"(*(தசம (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        திரும்பு 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    மாறிலி int64_t elems_per_cl = 16;
    மாறிலி int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    மாறிலி int64_t ne1_data_end = ne1 - rp1;
    மாறிலி int64_t ne2_data_end = ne2 - rp2;
    மாறிலி int64_t ne3_data_end = ne3 - rp3;

    க்கு (int64_t cl = thread_id; cl < total_cl; cl += இலக்கம்_threads) {
        மாறிலி int64_t elem_start = cl * elems_per_cl;
        int64_t       elem_end   = elem_start + elems_per_cl;
        எனில் (elem_end > total_elements) {
            elem_end = total_elements;
        }

        க்கு (int64_t idx = elem_start; idx < elem_end; idx++) {
            மாறிலி int64_t i0   = idx % ne0;
            மாறிலி int64_t rem  = idx / ne0;
            மாறிலி int64_t i1   = rem % ne1;
            மாறிலி int64_t rem2 = rem / ne1;
            மாறிலி int64_t i2   = rem2 % ne2;
            மாறிலி int64_t i3   = rem2 / ne2;

            எனில் (i1 >= lp1 && i1 < ne1_data_end && i2 >= lp2 && i2 < ne2_data_end && i3 >= lp3 && i3 < ne3_data_end) {
                மாறிலி தசம * sp = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_data[idx]    = *sp;
            } இல்லையெனில் {
                dst_data[idx] = 0.0f;
            }
        }
    }

    திரும்பு 0;
}
