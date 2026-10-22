//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_sum_rows_params {
    ସଂରଚନା ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ସଂରଚନା ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_sum_rows_params * params, ଶୂନ୍ୟ * env) {
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

    ଦଶମିକ * src0_data = (ଦଶମିକ *) src0->data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }


    ଅଚଳ int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    ଅଚଳ int64_t ne01 = src0->ne[1];
    ଅଚଳ int64_t ne02 = src0->ne[2];
    ଅଚଳ int64_t ne03 = src0->ne[3];

    ଅଚଳ size_t nb01 = src0->nb[1];
    ଅଚଳ size_t nb02 = src0->nb[2];
    ଅଚଳ size_t nb03 = src0->nb[3];

    ଅଚଳ size_t nb1 = dst->nb[1];
    ଅଚଳ size_t nb2 = dst->nb[2];
    ଅଚଳ size_t nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    ଅଚଳ int64_t total_rows = ne01 * ne02 * ne03;

    ପାଇଁ (int64_t ir = thread_id; ir < total_rows; ir += ଅଙ୍କ_threads) {
        ଅଚଳ int64_t i03 = ir / (ne02 * ne01);
        ଅଚଳ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ଅଚଳ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ଦଶମିକ *       dst_ptr = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        ଦଶମିକ zero = 0.0f;
        __asm__ ଅସ୍ଥିର("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ପାଇଁ (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ ଅସ୍ଥିର(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & src_row[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        ଦଶମିକ row_sum;
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

    ଫେରନ୍ତୁ 0;
}
