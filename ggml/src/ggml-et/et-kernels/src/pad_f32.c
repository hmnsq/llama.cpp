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

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

সংরচনা ggml_et_pad_params {
    সংরচনা ggml_tensor src0;
    সংরচনা ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
স্থির অন্তঃস্থ শূন্য vec_copy_f32(ভিন্ন * dst, ধ্রুব ভিন্ন * src, int32_t n) {
    int32_t       i       = 0;
    ধ্রুব int32_t vec_end = (n / 8) * 8;
    প্রতি (; i < vec_end; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [s] "m"(*(ধ্রুব ভিন্ন (*)[8]) & src[i])
            : "f10");
    }
    প্রতি (; i < n; i++) {
        dst[i] = src[i];
    }
}

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_pad_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        প্রত্যাগম -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        প্রত্যাগম 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        প্রত্যাগম -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;
    }

    ধ্রুব ভিন্ন * src0_data = (ধ্রুব ভিন্ন *) src0->data;
    ভিন্ন *       dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        প্রত্যাগম -1;
    }

    // Dst dimensions
    ধ্রুব int64_t ne0 = dst->ne[0];
    ধ্রুব int64_t ne1 = dst->ne[1];
    ধ্রুব int64_t ne2 = dst->ne[2];
    ধ্রুব int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    ধ্রুব int64_t nb1_src = src0->nb[1];
    ধ্রুব int64_t nb2_src = src0->nb[2];
    ধ্রুব int64_t nb3_src = src0->nb[3];

    // Padding values
    ধ্রুব int32_t lp1 = params->lp[1];
    ধ্রুব int32_t rp1 = params->rp[1];
    ধ্রুব int32_t lp2 = params->lp[2];
    ধ্রুব int32_t rp2 = params->rp[2];
    ধ্রুব int32_t lp3 = params->lp[3];
    ধ্রুব int32_t rp3 = params->rp[3];

    ধ্রুব int64_t total_rows     = ne1 * ne2 * ne3;
    ধ্রুব int64_t total_elements = ne0 * total_rows;

    যদি (total_elements == 0) {
        প্রত্যাগম 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    ভিন্ন zero = 0.0f;
    __asm__ চঞ্চল("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    যদি (ne0 % 16 == 0) {
        প্রতি (int64_t row = thread_id; row < total_rows; row += অঙ্ক_threads) {
            ধ্রুব int64_t i3 = row / (ne1 * ne2);
            ধ্রুব int64_t i2 = (row / ne1) % ne2;
            ধ্রুব int64_t i1 = row % ne1;

            ভিন্ন * dst_row = dst_data + row * ne0;

            যদি (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                ধ্রুব ভিন্ন * src_row = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_row, src_row, (int32_t) ne0);
            } অন্যথা {
                int64_t       i       = 0;
                ধ্রুব int64_t vec_end = (ne0 / 8) * 8;
                প্রতি (; i < vec_end; i += 8) {
                    __asm__ চঞ্চল("fsw.ps f12, %[d]\n" : [d] "=m"(*(ভিন্ন (*)[8]) & dst_row[i])::"f12");
                }
            }
        }
        প্রত্যাগম 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    ধ্রুব int64_t elems_per_cl = 16;
    ধ্রুব int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    ধ্রুব int64_t ne1_data_end = ne1 - rp1;
    ধ্রুব int64_t ne2_data_end = ne2 - rp2;
    ধ্রুব int64_t ne3_data_end = ne3 - rp3;

    প্রতি (int64_t cl = thread_id; cl < total_cl; cl += অঙ্ক_threads) {
        ধ্রুব int64_t elem_start = cl * elems_per_cl;
        int64_t       elem_end   = elem_start + elems_per_cl;
        যদি (elem_end > total_elements) {
            elem_end = total_elements;
        }

        প্রতি (int64_t idx = elem_start; idx < elem_end; idx++) {
            ধ্রুব int64_t i0   = idx % ne0;
            ধ্রুব int64_t rem  = idx / ne0;
            ধ্রুব int64_t i1   = rem % ne1;
            ধ্রুব int64_t rem2 = rem / ne1;
            ধ্রুব int64_t i2   = rem2 % ne2;
            ধ্রুব int64_t i3   = rem2 / ne2;

            যদি (i1 >= lp1 && i1 < ne1_data_end && i2 >= lp2 && i2 < ne2_data_end && i3 >= lp3 && i3 < ne3_data_end) {
                ধ্রুব ভিন্ন * sp = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_data[idx]    = *sp;
            } অন্যথা {
                dst_data[idx] = 0.0f;
            }
        }
    }

    প্রত্যাগম 0;
}
