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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_pad_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
સ્થિર અંતઃરેખ શૂન્ય vec_copy_f32(દશાંશ * dst, અચલ દશાંશ * src, int32_t n) {
    int32_t       i       = 0;
    અચલ int32_t vec_end = (n / 8) * 8;
    માટે (; i < vec_end; i += 8) {
        __asm__ અસ્થિર(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(દશાંશ (*)[8]) & dst[i])
            : [s] "m"(*(અચલ દશાંશ (*)[8]) & src[i])
            : "f10");
    }
    માટે (; i < n; i++) {
        dst[i] = src[i];
    }
}

પૂર્ણાંક entry_point(સંરચના ggml_et_pad_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    અચલ દશાંશ * src0_data = (અચલ દશાંશ *) src0->data;
    દશાંશ *       dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !dst_data) {
        પાછા -1;
    }

    // Dst dimensions
    અચલ int64_t ne0 = dst->ne[0];
    અચલ int64_t ne1 = dst->ne[1];
    અચલ int64_t ne2 = dst->ne[2];
    અચલ int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    અચલ int64_t nb1_src = src0->nb[1];
    અચલ int64_t nb2_src = src0->nb[2];
    અચલ int64_t nb3_src = src0->nb[3];

    // Padding values
    અચલ int32_t lp1 = params->lp[1];
    અચલ int32_t rp1 = params->rp[1];
    અચલ int32_t lp2 = params->lp[2];
    અચલ int32_t rp2 = params->rp[2];
    અચલ int32_t lp3 = params->lp[3];
    અચલ int32_t rp3 = params->rp[3];

    અચલ int64_t total_rows     = ne1 * ne2 * ne3;
    અચલ int64_t total_elements = ne0 * total_rows;

    જો (total_elements == 0) {
        પાછા 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    દશાંશ zero = 0.0f;
    __asm__ અસ્થિર("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    જો (ne0 % 16 == 0) {
        માટે (int64_t row = thread_id; row < total_rows; row += અંક_threads) {
            અચલ int64_t i3 = row / (ne1 * ne2);
            અચલ int64_t i2 = (row / ne1) % ne2;
            અચલ int64_t i1 = row % ne1;

            દશાંશ * dst_row = dst_data + row * ne0;

            જો (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } નહીંતર {
                int64_t       i       = 0;
                અચલ int64_t vec_end = (ne0 / 8) * 8;
                માટે (; i < vec_end; i += 8) {
                    __asm__ અસ્થિર("fsw.ps f12, %[d]\n" : [d] "=m"(*(દશાંશ (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        પાછા 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    અચલ int64_t elems_per_cl = 16;
    અચલ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    અચલ int64_t ne1_data_end = ne1 - rp1;
    અચલ int64_t ne2_data_end = ne2 - rp2;
    અચલ int64_t ne3_data_end = ne3 - rp3;

    માટે (int64_t cl = thread_id; cl < total_cl; cl += અંક_threads) {
        અચલ int64_t elem_start = cl * elems_per_cl;
        int64_t       elem_end   = elem_start + elems_per_cl;
        જો (elem_end > total_elements) {
            elem_end = total_elements;
        }

        માટે (int64_t idx = elem_start; idx < elem_end; idx++) {
            અચલ int64_t i0   = idx % ne0;
            અચલ int64_t rem  = idx / ne0;
            અચલ int64_t i1   = rem % ne1;
            અચલ int64_t rem2 = rem / ne1;
            અચલ int64_t i2   = rem2 % ne2;
            અચલ int64_t i3   = rem2 / ne2;

            જો (i1 >= lp1 && i1 < ne1_data_end && i2 >= lp2 && i2 < ne2_data_end && i3 >= lp3 && i3 < ne3_data_end) {
                અચલ દશાંશ * sp = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_data[idx]    = *sp;
            } નહીંતર {
                dst_data[idx] = 0.0f;
            }
        }
    }

    પાછા 0;
}
