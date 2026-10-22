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

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_pad_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
अटल अंतरभूत व्योम vec_copy_f32(भग्न * dst, स्थिर भग्न * src, int32_t n) {
    int32_t       i       = 0;
    स्थिर int32_t vec_end = (n / 8) * 8;
    क्रम (; i < vec_end; i += 8) {
        __asm__ अस्थिर(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(भग्न (*)[8]) & dst[i])
            : [s] "m"(*(स्थिर भग्न (*)[8]) & src[i])
            : "f10");
    }
    क्रम (; i < n; i++) {
        dst[i] = src[i];
    }
}

पूर्णांक entry_point(काष्ठा ggml_et_pad_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    स्थिर भग्न * src0_तथ्य = (स्थिर भग्न *) src0->तथ्य;
    भग्न *       dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    // Dst dimensions
    स्थिर int64_t ne0 = dst->ne[0];
    स्थिर int64_t ne1 = dst->ne[1];
    स्थिर int64_t ne2 = dst->ne[2];
    स्थिर int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    स्थिर int64_t nb1_src = src0->nb[1];
    स्थिर int64_t nb2_src = src0->nb[2];
    स्थिर int64_t nb3_src = src0->nb[3];

    // Padding values
    स्थिर int32_t lp1 = params->lp[1];
    स्थिर int32_t rp1 = params->rp[1];
    स्थिर int32_t lp2 = params->lp[2];
    स्थिर int32_t rp2 = params->rp[2];
    स्थिर int32_t lp3 = params->lp[3];
    स्थिर int32_t rp3 = params->rp[3];

    स्थिर int64_t योग_rows     = ne1 * ne2 * ne3;
    स्थिर int64_t योग_elements = ne0 * योग_rows;

    अगर (योग_elements == 0) {
        वापस 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    भग्न zero = 0.0f;
    __asm__ अस्थिर("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    अगर (ne0 % 16 == 0) {
        क्रम (int64_t row = thread_id; row < योग_rows; row += अंक_threads) {
            स्थिर int64_t i3 = row / (ne1 * ne2);
            स्थिर int64_t i2 = (row / ne1) % ne2;
            स्थिर int64_t i1 = row % ne1;

            भग्न * dst_row = dst_तथ्य + row * ne0;

            अगर (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } अन्यथा {
                int64_t       i       = 0;
                स्थिर int64_t vec_end = (ne0 / 8) * 8;
                क्रम (; i < vec_end; i += 8) {
                    __asm__ अस्थिर("fsw.ps f12, %[d]\n" : [d] "=m"(*(भग्न (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        वापस 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    स्थिर int64_t elems_per_cl = 16;
    स्थिर int64_t योग_cl     = (योग_elements + elems_per_cl - 1) / elems_per_cl;

    स्थिर int64_t ne1_तथ्य_end = ne1 - rp1;
    स्थिर int64_t ne2_तथ्य_end = ne2 - rp2;
    स्थिर int64_t ne3_तथ्य_end = ne3 - rp3;

    क्रम (int64_t cl = thread_id; cl < योग_cl; cl += अंक_threads) {
        स्थिर int64_t elem_start = cl * elems_per_cl;
        int64_t       elem_end   = elem_start + elems_per_cl;
        अगर (elem_end > योग_elements) {
            elem_end = योग_elements;
        }

        क्रम (int64_t idx = elem_start; idx < elem_end; idx++) {
            स्थिर int64_t i0   = idx % ne0;
            स्थिर int64_t rem  = idx / ne0;
            स्थिर int64_t i1   = rem % ne1;
            स्थिर int64_t rem2 = rem / ne1;
            स्थिर int64_t i2   = rem2 % ne2;
            स्थिर int64_t i3   = rem2 / ne2;

            अगर (i1 >= lp1 && i1 < ne1_तथ्य_end && i2 >= lp2 && i2 < ne2_तथ्य_end && i3 >= lp3 && i3 < ne3_तथ्य_end) {
                स्थिर भग्न * sp = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_तथ्य[idx]    = *sp;
            } अन्यथा {
                dst_तथ्य[idx] = 0.0f;
            }
        }
    }

    वापस 0;
}
