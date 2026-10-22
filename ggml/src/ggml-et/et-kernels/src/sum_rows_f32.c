//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_sum_rows_params {
    கட்டமைப்பு ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    கட்டமைப்பு ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

முழு entry_point(கட்டமைப்பு ggml_et_sum_rows_params * params, வெறுமை * env) {
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

    தசம * src0_data = (தசம *) src0->data;
    தசம * dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;
    }


    மாறிலி int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    மாறிலி int64_t ne01 = src0->ne[1];
    மாறிலி int64_t ne02 = src0->ne[2];
    மாறிலி int64_t ne03 = src0->ne[3];

    மாறிலி size_t nb01 = src0->nb[1];
    மாறிலி size_t nb02 = src0->nb[2];
    மாறிலி size_t nb03 = src0->nb[3];

    மாறிலி size_t nb1 = dst->nb[1];
    மாறிலி size_t nb2 = dst->nb[2];
    மாறிலி size_t nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    மாறிலி int64_t total_rows = ne01 * ne02 * ne03;

    க்கு (int64_t ir = thread_id; ir < total_rows; ir += இலக்கம்_threads) {
        மாறிலி int64_t i03 = ir / (ne02 * ne01);
        மாறிலி int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        மாறிலி int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        மாறிலி தசம * src_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        தசம *       dst_ptr = (தசம *) ((எழுத்து *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        தசம zero = 0.0f;
        __asm__ நிலையற்ற("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        க்கு (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ நிலையற்ற(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(மாறிலி தசம (*)[8]) & src_row[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        தசம row_sum;
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

    திரும்பு 0;
}
