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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_pad_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
स्थिर अन्तःस्थ शून्य vec_copy_f32(भिन्न * dst, ध्रुव भिन्न * src, int32_t n) {
    int32_t       i       = 0;
    ध्रुव int32_t vec_अन्तः = (n / 8) * 8;
    प्रति (; i < vec_अन्तः; i += 8) {
        __asm__ चञ्चल(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(भिन्न (*)[8]) & dst[i])
            : [s] "m"(*(ध्रुव भिन्न (*)[8]) & src[i])
            : "f10");
    }
    प्रति (; i < n; i++) {
        dst[i] = src[i];
    }
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_pad_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न * src0_दत्त = (ध्रुव भिन्न *) src0->दत्त;
    भिन्न *       dst_दत्त  = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }

    // Dst dimensions
    ध्रुव int64_t ne0 = dst->ne[0];
    ध्रुव int64_t ne1 = dst->ne[1];
    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    ध्रुव int64_t nb1_src = src0->nb[1];
    ध्रुव int64_t nb2_src = src0->nb[2];
    ध्रुव int64_t nb3_src = src0->nb[3];

    // Padding values
    ध्रुव int32_t lp1 = params->lp[1];
    ध्रुव int32_t rp1 = params->rp[1];
    ध्रुव int32_t lp2 = params->lp[2];
    ध्रुव int32_t rp2 = params->rp[2];
    ध्रुव int32_t lp3 = params->lp[3];
    ध्रुव int32_t rp3 = params->rp[3];

    ध्रुव int64_t योग_पङ्क्तिकाः     = ne1 * ne2 * ne3;
    ध्रुव int64_t योग_elements = ne0 * योग_पङ्क्तिकाः;

    यदि (योग_elements == 0) {
        प्रत्यागम 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    भिन्न zero = 0.0f;
    __asm__ चञ्चल("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    यदि (ne0 % 16 == 0) {
        प्रति (int64_t row = thread_id; row < योग_पङ्क्तिकाः; row += अङ्कः_तन्तवः) {
            ध्रुव int64_t i3 = row / (ne1 * ne2);
            ध्रुव int64_t i2 = (row / ne1) % ne2;
            ध्रुव int64_t i1 = row % ne1;

            भिन्न * dst_row = dst_दत्त + row * ne0;

            यदि (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } अन्यथा {
                int64_t       i       = 0;
                ध्रुव int64_t vec_अन्तः = (ne0 / 8) * 8;
                प्रति (; i < vec_अन्तः; i += 8) {
                    __asm__ चञ्चल("fsw.ps f12, %[d]\n" : [d] "=m"(*(भिन्न (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        प्रत्यागम 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    ध्रुव int64_t elems_प्रत्येकम्_cl = 16;
    ध्रुव int64_t योग_cl     = (योग_elements + elems_प्रत्येकम्_cl - 1) / elems_प्रत्येकम्_cl;

    ध्रुव int64_t ne1_दत्त_अन्तः = ne1 - rp1;
    ध्रुव int64_t ne2_दत्त_अन्तः = ne2 - rp2;
    ध्रुव int64_t ne3_दत्त_अन्तः = ne3 - rp3;

    प्रति (int64_t cl = thread_id; cl < योग_cl; cl += अङ्कः_तन्तवः) {
        ध्रुव int64_t elem_start = cl * elems_प्रत्येकम्_cl;
        int64_t       elem_अन्तः   = elem_start + elems_प्रत्येकम्_cl;
        यदि (elem_अन्तः > योग_elements) {
            elem_अन्तः = योग_elements;
        }

        प्रति (int64_t idx = elem_start; idx < elem_अन्तः; idx++) {
            ध्रुव int64_t i0   = idx % ne0;
            ध्रुव int64_t rem  = idx / ne0;
            ध्रुव int64_t i1   = rem % ne1;
            ध्रुव int64_t rem2 = rem / ne1;
            ध्रुव int64_t i2   = rem2 % ne2;
            ध्रुव int64_t i3   = rem2 / ne2;

            यदि (i1 >= lp1 && i1 < ne1_दत्त_अन्तः && i2 >= lp2 && i2 < ne2_दत्त_अन्तः && i3 >= lp3 && i3 < ne3_दत्त_अन्तः) {
                ध्रुव भिन्न * sp = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_दत्त[idx]    = *sp;
            } अन्यथा {
                dst_दत्त[idx] = 0.0f;
            }
        }
    }

    प्रत्यागम 0;
}
