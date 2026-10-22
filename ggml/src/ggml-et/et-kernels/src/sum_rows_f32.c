//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

সংরচনা ggml_et_sum_rows_params {
    সংরচনা ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    সংরচনা ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_sum_rows_params * params, শূন্য * env) {
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

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        প্রত্যাগম -1;
    }


    ধ্রুব int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    ধ্রুব int64_t ne01 = src0->ne[1];
    ধ্রুব int64_t ne02 = src0->ne[2];
    ধ্রুব int64_t ne03 = src0->ne[3];

    ধ্রুব size_t nb01 = src0->nb[1];
    ধ্রুব size_t nb02 = src0->nb[2];
    ধ্রুব size_t nb03 = src0->nb[3];

    ধ্রুব size_t nb1 = dst->nb[1];
    ধ্রুব size_t nb2 = dst->nb[2];
    ধ্রুব size_t nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    ধ্রুব int64_t total_rows = ne01 * ne02 * ne03;

    প্রতি (int64_t ir = thread_id; ir < total_rows; ir += অঙ্ক_threads) {
        ধ্রুব int64_t i03 = ir / (ne02 * ne01);
        ধ্রুব int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ধ্রুব int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        ধ্রুব ভিন্ন * src_row = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ভিন্ন *       dst_ptr = (ভিন্ন *) ((বর্ণ *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        ভিন্ন zero = 0.0f;
        __asm__ চঞ্চল("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        প্রতি (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ চঞ্চল(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ধ্রুব ভিন্ন (*)[8]) & src_row[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        ভিন্ন row_sum;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(row_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

        atomic_store_f32(dst_ptr, row_sum);
    }

    প্রত্যাগম 0;
}
